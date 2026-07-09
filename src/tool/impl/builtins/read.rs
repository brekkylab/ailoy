use std::path::Path;

use crate::{
    datatype::Bytes,
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_CHARS: usize = 10000;
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;
const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

enum FileKind {
    Image(&'static str),
    Text,
}

fn classify(bytes: &[u8]) -> Result<FileKind, String> {
    let Some(kind) = infer::get(bytes) else {
        return Ok(FileKind::Text);
    };
    match kind.mime_type() {
        "image/png" => Ok(FileKind::Image("image/png")),
        "image/jpeg" => Ok(FileKind::Image("image/jpeg")),
        "image/gif" => Ok(FileKind::Image("image/gif")),
        "image/webp" => Ok(FileKind::Image("image/webp")),
        mime if mime.starts_with("text/") => Ok(FileKind::Text),
        mime => Err(format!("unsupported file type: {mime}")),
    }
}

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

pub fn get_read_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("read")
        .description(
            concat!(
                "Reads a file from the local filesystem. ",
                "It can read text files or images(PNG/JPEG/GIF/WEBP). ",
                "When you already know which part of the file you need, only read that part. This can be important for larger files. ",
                "For text files, results are returned using cat -n format, with line numbers starting at 1. ",
                "Lines longer than 10000 characters are truncated. ",
                "Binary or unsupported file types return an error. ",
            )
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once. Ignored for images.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once. Ignored for images.",
                    "default": 2000,
                }
            },
            "required": ["path"]
        }))
        .build()
}

pub fn get_read_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, console: &dyn Console| -> Message {
            let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
                return error_message(id, "missing required parameter: path", "validation");
            };
            let path = Path::new(path_str);

            let bytes = match console.read(path).await {
                Ok(b) => b,
                Err(e) => return error_message(id, format!("read {path_str}: {e}"), "io"),
            };

            let kind = match classify(&bytes) {
                Ok(k) => k,
                Err(e) => return error_message(id, e, "validation"),
            };

            match kind {
                FileKind::Image(mime) => {
                    if bytes.len() > MAX_IMAGE_BYTES {
                        return error_message(
                            id,
                            format!(
                                "image too large: {} bytes (limit: {})",
                                bytes.len(),
                                MAX_IMAGE_BYTES
                            ),
                            "validation",
                        );
                    }
                    let part = Part::image_embedded(mime, Bytes::from(bytes))
                        .expect("image_embedded always succeeds");
                    Message::new(Role::Tool).with_contents([part]).with_id(id)
                }
                FileKind::Text => {
                    if bytes.len() > MAX_FILE_BYTES {
                        return error_message(
                            id,
                            format!(
                                "file too large: {} bytes (limit: {}); use offset/limit to read in chunks",
                                bytes.len(),
                                MAX_FILE_BYTES
                            ),
                            "validation",
                        );
                    }
                    let (cow, _, _) = encoding_rs::UTF_8.decode(&bytes);
                    let text = cow.into_owned();
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
                    let (content, total) = format_text(&text, offset, limit);
                    Message::new(Role::Tool)
                        .with_contents([Part::value(crate::to_value!({
                            "content": content.as_str(),
                            "total_lines": total as i64,
                        }))])
                        .with_id(id)
                }
            }
        }
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        datatype::Value,
        runenv::{Local, Machine},
        to_value,
        tool::ToolProvider,
    };

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("read", get_read_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_read_tool_desc()]).unwrap();
        let f = funcs.get("read").unwrap();
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        f.call(args, "1", console).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_read_returns_numbered_lines() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "alpha\nbeta\ngamma\n").unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        let content = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/content")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string();
        assert!(content.contains("\talpha"), "got: {content}");
        assert!(content.contains("\tbeta"), "got: {content}");
        assert!(content.contains("\tgamma"), "got: {content}");
        assert!(
            content
                .lines()
                .next()
                .unwrap()
                .trim_start()
                .starts_with("1\t")
        );
    }

    #[tokio::test]
    async fn test_read_offset_and_limit() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\nb\nc\nd\ne\n").unwrap();
        let msg = call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "offset": 2,
            "limit": 2,
        }))
        .await;
        let content = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/content")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string();
        assert!(content.contains("\tb"));
        assert!(content.contains("\tc"));
        assert!(!content.contains("\ta"), "should skip first line");
        assert!(!content.contains("\td"), "should respect limit");
    }

    #[tokio::test]
    async fn test_read_missing_path() {
        let msg = call(to_value!({})).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }

    #[tokio::test]
    async fn test_read_nonexistent_returns_io_error() {
        let msg = call(to_value!({ "path": "/this/path/does/not/exist/xyz" })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "io");
    }

    #[tokio::test]
    async fn test_read_unsupported_binary() {
        // Real PDF magic bytes — classified as unsupported by content sniffing.
        let pdf: &[u8] = b"%PDF-1.4\n%\xC7\xEC\x8F\xA2\n1 0 obj\n<<>>\nendobj\n";
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), pdf).unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            err.contains("unsupported"),
            "expected unsupported error, got: {err}"
        );
    }

    fn read_content(msg: &Message) -> String {
        msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/content")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn test_read_utf8_bom_stripped() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice("hello\n".as_bytes());
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        let content = read_content(&msg);
        assert!(content.contains("\thello"), "got: {content}");
        assert!(!content.contains('\u{FEFF}'), "BOM should be stripped");
    }

    #[tokio::test]
    async fn test_read_utf16_le() {
        // UTF-16 LE BOM + "hi\n"
        let bytes: &[u8] = &[0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00, 0x0A, 0x00];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), bytes).unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        let content = read_content(&msg);
        assert!(content.contains("\thi"), "got: {content}");
    }

    #[tokio::test]
    async fn test_read_image_returns_image_part() {
        // Minimal 1x1 red PNG (valid PNG bytes)
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        std::fs::write(tmp.path(), png).unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        assert!(msg.contents[0].is_image(), "expected image part");
    }
}
