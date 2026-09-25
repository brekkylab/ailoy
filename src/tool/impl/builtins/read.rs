use super::imgread::image_mime;
use crate::{
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_CHARS: usize = 10000;
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

fn check_text(bytes: &[u8]) -> Result<(), String> {
    if image_mime(bytes).is_some() {
        return Err("file is an image; use `imgread` to read images".to_string());
    }
    match infer::get(bytes) {
        None => Ok(()),
        Some(kind) if kind.mime_type().starts_with("text/") => Ok(()),
        Some(kind) => Err(format!("unsupported file type: {}", kind.mime_type())),
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
                "Reads a text file from the local filesystem. ",
                "When you already know which part of the file you need, only read that part. This can be important for larger files. ",
                "Results are returned using cat -n format, with line numbers starting at 1. ",
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
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    "default": 2000,
                }
            },
            "required": ["path"]
        }))
        .build()
}

pub fn get_read_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, console: &mut Console| -> Message {
            let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
                return error_message(id, "missing required parameter: path", "validation");
            };

            // One `read`. `size` is the whole file's, so a short answer means the
            // file is bigger than one message — reported rather than returned as if
            // it were the file, which is what `MAX_FILE_BYTES` below also guards.
            let bytes = match console.read(path_str, None, None).await {
                Ok(r) if (r.data.len() as u64) < r.size => {
                    return error_message(
                        id,
                        format!(
                            "read {path_str}: file is {} bytes, more than one message carries",
                            r.size
                        ),
                        "io",
                    );
                }
                Ok(r) => r.data,
                Err(e) => return error_message(id, format!("read {path_str}: {e}"), "io"),
            };

            if let Err(e) = check_text(&bytes) {
                return error_message(id, e, "validation");
            }

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
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{datatype::Value, test_console, to_value, tool::ToolProvider};

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("read", get_read_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_read_tool_desc()]).unwrap();
        let f = funcs.get("read").unwrap();
        let mut console = test_console().await;
        f.call(args, "1", &mut console)
            .next()
            .await
            .unwrap()
            .message
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
    async fn test_read_image_points_to_imgread() {
        // PNG signature is enough for content sniffing.
        let png: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00];
        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        std::fs::write(tmp.path(), png).unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(err.contains("imgread"), "got: {err}");
    }
}
