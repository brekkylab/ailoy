use super::imgread::image_mime;
use crate::{
    console::Sandbox,
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

/// The text of the file at `path`, for both `read` styles. An error comes with
/// its phase, `"io"` or `"validation"`.
pub(super) async fn load_text(
    sandbox: &Sandbox,
    path: &str,
) -> Result<String, (String, &'static str)> {
    let bytes = match crate::console::read(sandbox, path).await {
        Ok(bytes) => bytes,
        Err(e) => return Err((format!("read {path}: {e}"), "io")),
    };

    check_text(&bytes).map_err(|e| (e, "validation"))?;

    if bytes.len() > MAX_FILE_BYTES {
        return Err((
            format!(
                "file too large: {} bytes (limit: {}); read it in chunks",
                bytes.len(),
                MAX_FILE_BYTES
            ),
            "validation",
        ));
    }
    let (cow, _, _) = encoding_rs::UTF_8.decode(&bytes);
    Ok(cow.into_owned())
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

/// `read` after Claude Code's `Read`: lines numbered `cat -n` style. See
/// [`super::get_read_file_tool_desc`] for Gemini CLI's.
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
        async |args: Value, id: String, sandbox: &Sandbox| -> Message {
            let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
                return error_message(id, "missing required parameter: path", "validation");
            };
            let text = match load_text(sandbox, path_str).await {
                Ok(text) => text,
                Err((e, phase)) => return error_message(id, e, phase),
            };
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
    use crate::{console::Sandbox, datatype::Value, test_sandbox, to_value};

    /// Where a test puts the file a call works on.
    const PATH: &str = "/tmp/subject";

    async fn call(sandbox: &Sandbox, args: Value) -> Message {
        let f = get_read_tool_func();
        f.call(args, "1", sandbox).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_read_returns_numbered_lines() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "alpha\nbeta\ngamma\n")
            .await
            .unwrap();
        let msg = call(&sandbox, to_value!({ "path": PATH })).await;
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
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "a\nb\nc\nd\ne\n")
            .await
            .unwrap();
        let msg = call(
            &sandbox,
            to_value!({
                "path": PATH,
                "offset": 2,
                "limit": 2,
            }),
        )
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
        let msg = call(&test_sandbox().await, to_value!({})).await;
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
        let msg = call(
            &test_sandbox().await,
            to_value!({ "path": "/this/path/does/not/exist/xyz" }),
        )
        .await;
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
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, pdf).await.unwrap();
        let msg = call(&sandbox, to_value!({ "path": PATH })).await;
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
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, &bytes).await.unwrap();
        let msg = call(&sandbox, to_value!({ "path": PATH })).await;
        let content = read_content(&msg);
        assert!(content.contains("\thello"), "got: {content}");
        assert!(!content.contains('\u{FEFF}'), "BOM should be stripped");
    }

    #[tokio::test]
    async fn test_read_utf16_le() {
        // UTF-16 LE BOM + "hi\n"
        let bytes: &[u8] = &[0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00, 0x0A, 0x00];
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, bytes).await.unwrap();
        let msg = call(&sandbox, to_value!({ "path": PATH })).await;
        let content = read_content(&msg);
        assert!(content.contains("\thi"), "got: {content}");
    }

    #[tokio::test]
    async fn test_read_image_points_to_imgread() {
        // PNG signature is enough for content sniffing.
        let png: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00];
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, png).await.unwrap();
        let msg = call(&sandbox, to_value!({ "path": PATH })).await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(err.contains("imgread"), "got: {err}");
    }
}
