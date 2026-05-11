use std::path::Path;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

const DESCRIPTION: &str = "Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.";

pub async fn build_write_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("write")
        .description(DESCRIPTION)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }))
        .build();

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let Some(path) = args.pointer("/file_path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: file_path",
                "phase": "validation"
            });
        };
        if !Path::new(path).is_absolute() {
            return crate::to_value!({
                "error": "file_path must be absolute",
                "phase": "validation"
            });
        }
        let Some(content) = args.pointer("/content").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: content",
                "phase": "validation"
            });
        };

        match ctx
            .runenv
            .write(Path::new(path), content.as_bytes())
            .await
        {
            Ok(()) => crate::to_value!({
                "ok": true,
                "bytes_written": content.len() as i64
            }),
            Err(e) => crate::to_value!({
                "error": format!("write {path}: {e}"),
                "phase": "io"
            }),
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
    async fn test_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        let tool = build_write_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": path.to_string_lossy().to_string(),
                    "content": "ailoy",
                }),
                local_ctx(),
            )
            .await;
        let ok = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(ok);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "ailoy");
    }

    #[tokio::test]
    async fn test_write_overwrites_existing() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "old").unwrap();
        let tool = build_write_tool().await.unwrap().make(&spec());
        tool.call_next(
            to_value!({
                "file_path": tmp.path().to_string_lossy().to_string(),
                "content": "new",
            }),
            local_ctx(),
        )
        .await;
        assert_eq!(std::fs::read_to_string(tmp.path()).unwrap(), "new");
    }

    #[tokio::test]
    async fn test_write_validation_errors() {
        let tool = build_write_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "content": "x" }), local_ctx())
            .await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }
}
