use xberg::core::config::{ExtractInput, ExtractInputKind, ExtractionConfig};

use crate::{
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_CHARS: usize = 10000;
// Documents carry their formatting with them, so the same prose is a much
// bigger file than source text — a slide deck or a scan is mostly images by
// byte count. The cap is on the file, not on what comes back: extraction
// throws most of those bytes away.
const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;

fn error_message(id: String, msg: impl Into<String>, phase: &str) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::value(crate::to_value!({
            "error": msg.into(),
            "phase": phase,
        }))])
        .with_id(id)
}

fn format_text(text: &str, offset: usize, limit: usize) -> (String, usize) {
    let total = text.lines().count();
    let mut out = String::new();
    for (idx, line) in text
        .lines()
        .enumerate()
        .skip(offset.saturating_sub(1))
        .take(limit)
    {
        let line_no = idx + 1;
        let display: String = if line.chars().count() > MAX_LINE_CHARS {
            let truncated: String = line.chars().take(MAX_LINE_CHARS).collect();
            format!("{truncated} ... [truncated]")
        } else {
            line.to_string()
        };
        out.push_str(&format!("{line_no:>6}\t{display}\n"));
    }
    (out, total)
}

pub fn get_docread_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("docread")
        .description(
            concat!(
                "Extracts the text of a document file, converting it to plain text. ",
                "Handles Word (.docx/.doc/.odt/.rtf), Excel (.xlsx/.xls/.ods/.csv), ",
                "PowerPoint (.pptx/.ppt/.odp), PDF, EPUB, Hangul (.hwp/.hwpx), and HTML. ",
                "Use this for documents; use `read` for source and text files, which it returns verbatim. ",
                "Results are returned using cat -n format, with line numbers starting at 1. ",
                "Line numbers count the extracted text, not the file — they do not address anything `edit` can change. ",
                "Layout, styling and images are dropped; tables come back as their cell text. ",
                "A PDF with no text layer (a scan) extracts nothing: this tool does not run OCR. ",
            )
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the document to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number of the extracted text to start reading from. Only provide if the document is too large to read at once.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the document is too large to read at once.",
                    "default": 2000,
                }
            },
            "required": ["path"]
        }))
        .build()
}

pub fn get_docread_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, console: &mut Console| -> Message {
            let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
                return error_message(id, "missing required parameter: path", "validation");
            };

            // Whole file or nothing: a short answer against `size` means it is
            // bigger than one message, and these formats cannot be extracted
            // from a prefix — a PDF's xref and a zip's central directory both
            // live at the tail, so a partial read is an unreadable file.
            let bytes = match console.read(path_str, None, None).await {
                Ok(r) if (r.data.len() as u64) < r.size => {
                    return error_message(
                        id,
                        format!(
                            "docread {path_str}: file is {} bytes, more than one message carries",
                            r.size
                        ),
                        "io",
                    );
                }
                Ok(r) => r.data,
                Err(e) => return error_message(id, format!("docread {path_str}: {e}"), "io"),
            };

            if bytes.len() > MAX_FILE_BYTES {
                return error_message(
                    id,
                    format!(
                        "document too large: {} bytes (limit: {})",
                        bytes.len(),
                        MAX_FILE_BYTES
                    ),
                    "validation",
                );
            }

            // No `mime_type`: xberg sniffs the content and falls back to the
            // filename, which is what gets HWP and the OOXML family right —
            // they are all containers a magic-byte check alone reads as CFB or
            // zip. The name is a hint, not the decision.
            let filename = path_str
                .rsplit(['/', '\\'])
                .next()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let input = ExtractInput {
                kind: ExtractInputKind::Bytes,
                bytes: Some(bytes),
                filename,
                ..Default::default()
            };
            // OCR is not compiled in, so this only makes the refusal explicit
            // rather than letting a scan take the auto-OCR path and fail deeper.
            let config = ExtractionConfig {
                disable_ocr: true,
                ..Default::default()
            };

            let extracted = match xberg::extract(input, &config).await {
                Ok(out) => out,
                Err(e) => {
                    return error_message(id, format!("docread {path_str}: {e}"), "extract");
                }
            };
            let Some(doc) = extracted.results.into_iter().next() else {
                let why = extracted
                    .errors
                    .first()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "no content extracted".to_string());
                return error_message(id, format!("docread {path_str}: {why}"), "extract");
            };

            if doc.content.trim().is_empty() {
                return error_message(
                    id,
                    format!(
                        "docread {path_str}: {} holds no extractable text; if it is a scan, it needs OCR, which this tool does not run",
                        doc.mime_type
                    ),
                    "extract",
                );
            }

            let offset = args
                .pointer("/offset")
                .and_then(|v| v.as_integer())
                .map(|n| n.max(0) as usize)
                .unwrap_or(0);
            let limit = args
                .pointer("/limit")
                .and_then(|v| v.as_integer())
                .map(|n| n.max(0) as usize)
                .unwrap_or(DEFAULT_LIMIT);
            let (content, total) = format_text(&doc.content, offset, limit);

            Message::new(Role::Tool)
                .with_contents([Part::value(crate::to_value!({
                    "content": content.as_str(),
                    "total_lines": total as i64,
                    "mime_type": doc.mime_type.as_ref(),
                    "pages": doc.counts.pages as i64,
                    "tables": doc.counts.tables as i64,
                }))])
                .with_id(id)
        }
    )
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use futures::StreamExt;
    use zip::write::SimpleFileOptions;

    use super::*;
    use crate::{datatype::Value, test_console, to_value, tool::ToolProvider};

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("docread", get_docread_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_docread_tool_desc()]).unwrap();
        let f = funcs.get("docread").unwrap();
        let mut console = test_console().await;
        f.call(args, "1", &mut console)
            .next()
            .await
            .unwrap()
            .message
    }

    fn field<'a>(msg: &'a Message, key: &str) -> Option<&'a Value> {
        msg.contents[0]
            .as_value()
            .unwrap()
            .pointer(&format!("/{key}"))
    }

    fn text<'a>(msg: &'a Message, key: &str) -> &'a str {
        field(msg, key)
            .unwrap_or_else(|| panic!("no `{key}` in {:?}", msg.contents[0]))
            .as_str()
            .unwrap()
    }

    /// Write `bytes` under `name` in a fresh temp dir and call `docread` on it.
    async fn call_file(name: &str, bytes: &[u8]) -> Message {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        call(to_value!({ "path": path.to_string_lossy().to_string() })).await
    }

    /// The same, windowed to `limit` lines starting at `offset`.
    async fn call_file_window(name: &str, bytes: &[u8], offset: i64, limit: i64) -> Message {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        call(to_value!({
            "path": path.to_string_lossy().to_string(),
            "offset": offset,
            "limit": limit,
        }))
        .await
    }

    fn zip_of(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, data) in entries {
            // EPUB is a zip that is told apart from any other zip by a
            // `mimetype` member stored uncompressed at a fixed offset, so a
            // deflated one sniffs as `application/zip` and routes nowhere.
            // Callers pass it first; storing it is the other half.
            let options = if *name == "mimetype" {
                SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored)
            } else {
                SimpleFileOptions::default()
            };
            w.start_file(*name, options).unwrap();
            w.write_all(data.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    /// A structurally valid PDF with one page and no content stream: a parser
    /// reads it, and finds no text — the shape of a scan, without the scan.
    fn textless_pdf() -> Vec<u8> {
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
        }
        let xref_at = pdf.len();
        pdf.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
        // Every entry is exactly 20 bytes wide; the table is addressed by
        // multiplication, so a byte off here is an unreadable file.
        pdf.push_str("0000000000 65535 f \n");
        for at in &offsets {
            pdf.push_str(&format!("{at:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.into_bytes()
    }

    /// The three parts every OOXML package opens with: the content types, the
    /// root relationship, and `part` at `part_name`. Enough for a backend to
    /// route on, which keeps the fixtures readable XML rather than binaries.
    fn ooxml_entries(content_type: &str, part_name: &str, part: &str) -> Vec<(String, String)> {
        let types = format!(
            "{}{}{}<Override PartName=\"/{part_name}\" ContentType=\"{content_type}\"/></Types>",
            r#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">"#,
            r#"<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>"#,
            r#"<Default Extension="xml" ContentType="application/xml"/>"#,
        );
        let rel_type =
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument";
        let rels = format!(
            "{}<Relationship Id=\"rId1\" Type=\"{rel_type}\" Target=\"{part_name}\"/></Relationships>",
            r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
        );
        vec![
            ("[Content_Types].xml".to_string(), types),
            ("_rels/.rels".to_string(), rels),
            (part_name.to_string(), part.to_string()),
        ]
    }

    fn zip_of_owned(entries: &[(String, String)]) -> Vec<u8> {
        zip_of(
            &entries
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_str()))
                .collect::<Vec<_>>(),
        )
    }

    fn docx() -> Vec<u8> {
        zip_of_owned(&ooxml_entries(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
            "word/document.xml",
            concat!(
                r#"<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>"#,
                r#"<w:p><w:r><w:t>Ailoy quarterly report</w:t></w:r></w:p>"#,
                r#"<w:p><w:r><w:t>한글 본문도 그대로 나옵니다.</w:t></w:r></w:p>"#,
                r#"</w:body></w:document>"#,
            ),
        ))
    }

    fn xlsx() -> Vec<u8> {
        let sheet = concat!(
            r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>"#,
            r#"<row r="1"><c r="A1" t="inlineStr"><is><t>Quarter</t></is></c><c r="B1" t="inlineStr"><is><t>Revenue</t></is></c></row>"#,
            r#"<row r="2"><c r="A2" t="inlineStr"><is><t>Q1</t></is></c><c r="B2"><v>1200</v></c></row>"#,
            r#"</sheetData></worksheet>"#,
        );
        let workbook = concat!(
            r#"<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" "#,
            r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
            r#"<sheets><sheet name="Revenue" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        );
        let mut entries = ooxml_entries(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
            "xl/workbook.xml",
            workbook,
        );
        entries.push((
            "xl/_rels/workbook.xml.rels".to_string(),
            concat!(
                r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
                r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
            )
            .to_string(),
        ));
        entries.push(("xl/worksheets/sheet1.xml".to_string(), sheet.to_string()));
        zip_of_owned(&entries)
    }

    fn pptx() -> Vec<u8> {
        let presentation = concat!(
            r#"<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" "#,
            r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
            r#"<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>"#,
        );
        let slide = concat!(
            r#"<?xml version="1.0"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" "#,
            r#"xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree>"#,
            r#"<p:sp><p:txBody><a:p><a:r><a:t>Ailoy roadmap</a:t></a:r></a:p></p:txBody></p:sp>"#,
            r#"</p:spTree></p:cSld></p:sld>"#,
        );
        let mut entries = ooxml_entries(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
            "ppt/presentation.xml",
            presentation,
        );
        entries.push((
            "ppt/_rels/presentation.xml.rels".to_string(),
            concat!(
                r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
                r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>"#,
            )
            .to_string(),
        ));
        entries.push(("ppt/slides/slide1.xml".to_string(), slide.to_string()));
        zip_of_owned(&entries)
    }

    fn epub() -> Vec<u8> {
        zip_of(&[
            ("mimetype", "application/epub+zip"),
            (
                "META-INF/container.xml",
                concat!(
                    r#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">"#,
                    r#"<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#,
                ),
            ),
            (
                "OEBPS/content.opf",
                concat!(
                    r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">"#,
                    r#"<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Ailoy Handbook</dc:title>"#,
                    r#"<dc:identifier id="id">urn:uuid:1</dc:identifier><dc:language>en</dc:language></metadata>"#,
                    r#"<manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>"#,
                    r#"<spine><itemref idref="c1"/></spine></package>"#,
                ),
            ),
            (
                "OEBPS/ch1.xhtml",
                concat!(
                    r#"<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter 1</title></head>"#,
                    r#"<body><h1>Chapter One</h1><p>An ebook paragraph.</p></body></html>"#,
                ),
            ),
        ])
    }

    #[tokio::test]
    async fn test_docread_docx_returns_numbered_text() {
        let msg = call_file("report.docx", &docx()).await;
        let content = text(&msg, "content");
        assert!(content.contains("Ailoy quarterly report"), "got: {content}");
        assert!(content.contains("한글 본문도"), "got: {content}");
        assert!(
            content
                .lines()
                .next()
                .unwrap()
                .trim_start()
                .starts_with("1\t"),
            "got: {content}"
        );
    }

    #[tokio::test]
    async fn test_docread_reports_source_format() {
        let msg = call_file("report.docx", &docx()).await;
        let mime = text(&msg, "mime_type");
        assert!(mime.contains("wordprocessingml"), "got: {mime}");
    }

    #[tokio::test]
    async fn test_docread_offset_and_limit() {
        // Which line the second paragraph lands on is the extractor's call, so
        // read the whole thing first and window onto what it actually produced.
        let whole = call_file("report.docx", &docx()).await;
        let line_no = text(&whole, "content")
            .lines()
            .find(|l| l.contains("한글 본문도"))
            .and_then(|l| l.trim_start().split('\t').next()?.parse::<i64>().ok())
            .expect("the second paragraph is numbered");
        assert!(line_no > 1, "fixture must have a line before it");

        let msg = call_file_window("report.docx", &docx(), line_no, 1).await;
        let content = text(&msg, "content");
        assert!(content.contains("한글 본문도"), "got: {content}");
        assert!(!content.contains("quarterly"), "should skip earlier lines");
        assert_eq!(content.lines().count(), 1, "limit is one line: {content}");
    }

    #[tokio::test]
    async fn test_docread_xlsx_returns_cell_text() {
        let msg = call_file("book.xlsx", &xlsx()).await;
        let content = text(&msg, "content");
        assert!(content.contains("Quarter"), "got: {content}");
        assert!(content.contains("1200"), "got: {content}");
    }

    #[tokio::test]
    async fn test_docread_pptx_returns_slide_text() {
        let msg = call_file("deck.pptx", &pptx()).await;
        let content = text(&msg, "content");
        assert!(content.contains("Ailoy roadmap"), "got: {content}");
    }

    #[tokio::test]
    async fn test_docread_epub_returns_chapter_text() {
        let msg = call_file("book.epub", &epub()).await;
        let content = text(&msg, "content");
        assert!(content.contains("Chapter One"), "got: {content}");
        assert!(content.contains("An ebook paragraph."), "got: {content}");
    }

    #[tokio::test]
    async fn test_docread_html_extracts_text_not_source() {
        let html = b"<!doctype html><html><body><h1>Heading</h1><p>Body text.</p></body></html>";
        let msg = call_file("page.html", html).await;
        let content = text(&msg, "content");
        assert!(content.contains("Heading"), "got: {content}");
        assert!(content.contains("Body text."), "got: {content}");
        assert!(
            !content.contains("<h1>"),
            "markup should be gone: {content}"
        );
    }

    #[tokio::test]
    async fn test_docread_pdf_without_text_layer_says_so() {
        let msg = call_file("scan.pdf", &textless_pdf()).await;
        let err = text(&msg, "error");
        assert!(err.contains("OCR"), "got: {err}");
        assert_eq!(text(&msg, "phase"), "extract");
    }

    #[tokio::test]
    async fn test_docread_missing_path() {
        let msg = call(to_value!({})).await;
        assert_eq!(text(&msg, "phase"), "validation");
    }

    #[tokio::test]
    async fn test_docread_nonexistent_returns_io_error() {
        let msg = call(to_value!({ "path": "/this/path/does/not/exist/xyz.docx" })).await;
        assert_eq!(text(&msg, "phase"), "io");
    }
}
