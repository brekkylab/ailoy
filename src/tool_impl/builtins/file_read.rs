use std::path::Path;

use crate::{
    datatype::{Bytes, Value},
    message::{Message, Part, Role, ToolDescBuilder},
    to_value,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

enum FileKind {
    Image(&'static str),
    Text,
}

fn error_message(id: String, msg: impl Into<String>) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::value(to_value!({"error": msg.into()}))])
        .with_id(id)
}

fn apply_char_window(s: String, offset: usize, limit: Option<usize>) -> String {
    if offset == 0 && limit.is_none() {
        return s;
    }
    let chars: Vec<char> = s.chars().collect();
    let start = offset.min(chars.len());
    let end = match limit {
        Some(n) => (start + n).min(chars.len()),
        None => chars.len(),
    };
    chars[start..end].iter().collect()
}

pub async fn build_file_read_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("file_read")
        .description(
            "Reads a file from the filesystem. Supports text files (returned as plain text) \
             and image files (PNG/JPEG/GIF/WEBP, returned as an image the model can see). \
             Binary or unsupported file types return an error. Paths must be absolute.",
        )
        .parameters(to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from (text files only). Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of characters to read (text files only). Optional."
                }
            },
            "required": ["path"]
        }))
        .build();

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let id = ctx.id.clone();

        let path_str = match args.pointer("/path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return error_message(id, "missing required parameter: path"),
        };
        let path = Path::new(&path_str);

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase());

        let kind = match ext.as_deref() {
            Some("png") => FileKind::Image("image/png"),
            Some("jpg") | Some("jpeg") => FileKind::Image("image/jpeg"),
            Some("gif") => FileKind::Image("image/gif"),
            Some("webp") => FileKind::Image("image/webp"),
            Some(
                e @ ("pdf" | "ipynb" | "zip" | "tar" | "gz" | "exe" | "bin" | "so" | "dylib"
                | "wasm"),
            ) => return error_message(id, format!("unsupported file type: .{e}")),
            _ => FileKind::Text,
        };

        let bytes = match ctx.runenv.read(path).await {
            Ok(b) => b,
            Err(e) => return error_message(id, format!("read failed: {e}")),
        };

        match kind {
            FileKind::Image(mime) => {
                let part = Part::image_embedded(mime, Bytes::from(bytes))
                    .expect("image_embedded always succeeds");
                Message::new(Role::Tool).with_contents([part]).with_id(id)
            }
            FileKind::Text => {
                let s = match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => {
                        return error_message(
                            id,
                            "file is not valid UTF-8; use bash to inspect binary files",
                        );
                    }
                };
                let offset = args
                    .pointer("/offset")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0)
                    .max(0) as usize;
                let limit = args
                    .pointer("/limit")
                    .and_then(|v| v.as_integer())
                    .map(|n| n.max(0) as usize);
                let content = apply_char_window(s, offset, limit);
                Message::new(Role::Tool)
                    .with_contents([Part::text(content)])
                    .with_id(id)
            }
        }
    });

    Ok(ToolFactory::simple(desc, f))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{agent::AgentSpec, runenv::Local, to_value, tool::ToolContext};

    fn spec() -> AgentSpec {
        AgentSpec::new("test")
    }

    fn local_ctx() -> ToolContext {
        ToolContext::new(String::new(), Arc::new(Local {}))
    }

    #[tokio::test]
    async fn test_missing_path_arg() {
        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool.call_next(to_value!({}), local_ctx()).await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            err.contains("missing"),
            "expected missing error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_nonexistent_path() {
        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({"path": "/tmp/ailoy_nonexistent_file_xyz.txt"}),
                local_ctx(),
            )
            .await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            err.contains("read failed"),
            "expected read error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_unsupported_extension() {
        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({"path": "/tmp/test.pdf"}), local_ctx())
            .await;
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

    #[tokio::test]
    async fn test_text_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello\nworld").unwrap();

        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({"path": tmp.path().to_str().unwrap()}),
                local_ctx(),
            )
            .await;
        let text = msg.contents[0].as_text().unwrap();
        assert_eq!(text, "hello\nworld");
    }

    #[tokio::test]
    async fn test_text_offset_limit() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "abcdefghij").unwrap();

        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({"path": tmp.path().to_str().unwrap(), "offset": 2, "limit": 4}),
                local_ctx(),
            )
            .await;
        let text = msg.contents[0].as_text().unwrap();
        assert_eq!(text, "cdef");
    }

    #[tokio::test]
    async fn test_image_png() {
        // Minimal 1x1 red PNG (valid PNG bytes)
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // width=1, height=1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // bit depth=8, color=RGB
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IHDR CRC + IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // IDAT data
            0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, // IDAT data + CRC
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82, // IEND CRC
        ];

        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        std::fs::write(tmp.path(), png).unwrap();

        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({"path": tmp.path().to_str().unwrap()}),
                local_ctx(),
            )
            .await;
        assert!(msg.contents[0].is_image(), "expected image part");
    }

    #[tokio::test]
    async fn test_invalid_utf8_text() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &[0xFF, 0xFE, 0x00]).unwrap();

        let tool = build_file_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({"path": tmp.path().to_str().unwrap()}),
                local_ctx(),
            )
            .await;
        let err = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(err.contains("UTF-8"), "expected UTF-8 error, got: {err}");
    }
}
