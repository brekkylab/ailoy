use crate::{
    datatype::Bytes,
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// Sniffs `bytes` for an image type the model can take. `None` when it is not
/// one of PNG/JPEG/GIF/WEBP.
pub(super) fn image_mime(bytes: &[u8]) -> Option<&'static str> {
    match infer::get(bytes)?.mime_type() {
        "image/png" => Some("image/png"),
        "image/jpeg" => Some("image/jpeg"),
        "image/gif" => Some("image/gif"),
        "image/webp" => Some("image/webp"),
        _ => None,
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

pub fn get_imgread_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("imgread")
        .description(concat!(
            "Reads an image file (PNG/JPEG/GIF/WEBP) from the local filesystem and returns it as an image. ",
            "Images larger than 5MB return an error. ",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the image file to read"
                }
            },
            "required": ["path"]
        }))
        .build()
}

pub fn get_imgread_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, console: &mut Console| -> Message {
            let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
                return error_message(id, "missing required parameter: path", "validation");
            };

            // Same single-`read` contract as the `read` tool: a short answer means
            // the file is bigger than one message carries.
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

            let Some(mime) = image_mime(&bytes) else {
                let found = infer::get(&bytes)
                    .map(|k| k.mime_type())
                    .unwrap_or("unknown");
                return error_message(
                    id,
                    format!("unsupported image type: {found}; use `read` for text files"),
                    "validation",
                );
            };

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
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{datatype::Value, test_console, to_value, tool::ToolProvider};

    async fn call(args: Value) -> Message {
        let mut provider = ToolProvider::new();
        provider.insert_func("imgread", get_imgread_tool_func());
        let funcs = provider.provide(&[get_imgread_tool_desc()]).unwrap();
        let f = funcs.get("imgread").unwrap();
        let mut console = test_console().await;
        f.call(args, "1", &mut console)
            .next()
            .await
            .unwrap()
            .message
    }

    fn phase(msg: &Message) -> String {
        msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn test_imgread_returns_image_part() {
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

    #[tokio::test]
    async fn test_imgread_rejects_text() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello\n").unwrap();
        let msg = call(to_value!({ "path": tmp.path().to_string_lossy().to_string() })).await;
        assert_eq!(phase(&msg), "validation");
    }

    #[tokio::test]
    async fn test_imgread_missing_path() {
        let msg = call(to_value!({})).await;
        assert_eq!(phase(&msg), "validation");
    }

    #[tokio::test]
    async fn test_imgread_nonexistent_returns_io_error() {
        let msg = call(to_value!({ "path": "/this/path/does/not/exist/xyz.png" })).await;
        assert_eq!(phase(&msg), "io");
    }
}
