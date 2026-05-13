use std::path::Path;

use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

pub fn get_write_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("write")
        .description(concat!(
            "Writes a text file to the local filesystem. ",
            "This tool will overwrite the existing file if there is one at the provided path."
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }))
        .build()
}

pub fn get_write_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, runenv: &dyn RunEnv| -> Value {
        let Some(path) = args.pointer("/path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: path",
                "phase": "validation",
            });
        };
        let Some(content) = args.pointer("/content").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: content",
                "phase": "validation",
            });
        };

        match runenv.write(Path::new(path), content.as_bytes()).await {
            Ok(()) => crate::to_value!({
                "ok": true,
                "bytes_written": content.len() as i64,
            }),
            Err(e) => crate::to_value!({
                "error": format!("write {path}: {e}"),
                "phase": "io",
            }),
        }
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{datatype::Value, message::Message, runenv::Local, to_value, tool::ToolProvider};

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("write", get_write_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_write_tool_desc()]).unwrap();
        let f = funcs.get("write").unwrap();
        f.call(args, "1", &Local {}).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_write_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        let msg = call(to_value!({
            "path": path.to_string_lossy().to_string(),
            "content": "ailoy",
        }))
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
        call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "content": "new",
        }))
        .await;
        assert_eq!(std::fs::read_to_string(tmp.path()).unwrap(), "new");
    }

    #[tokio::test]
    async fn test_write_validation_errors() {
        let msg = call(to_value!({ "content": "x" })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }
}
