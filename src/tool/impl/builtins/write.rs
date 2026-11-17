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
    tool_func!(async |args: Value, sandbox: &Sandbox| -> Value {
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

        // Its directory is made if it is not there.
        let wrote = crate::console::write(sandbox, path, content).await;

        match wrote {
            Ok(_) => crate::to_value!({
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
    use crate::{console::Sandbox, datatype::Value, message::Message, test_sandbox, to_value};

    /// Where a test puts the file a call works on.
    const PATH: &str = "/tmp/subject";

    async fn call(sandbox: &Sandbox, args: Value) -> Message {
        let f = get_write_tool_func();
        f.call(args, "1", sandbox).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_write_creates_file() {
        // In directories that are not there yet, which the tool makes.
        let path = "/tmp/new/dir/hello.txt";
        let sandbox = test_sandbox().await;
        let msg = call(
            &sandbox,
            to_value!({
                "path": path,
                "content": "ailoy",
            }),
        )
        .await;
        let ok = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(ok, "{msg:?}");
        assert_eq!(
            crate::console::read(&sandbox, path).await.unwrap(),
            b"ailoy"
        );
    }

    #[tokio::test]
    async fn test_write_overwrites_existing() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "old").await.unwrap();
        call(
            &sandbox,
            to_value!({
                "path": PATH,
                "content": "new",
            }),
        )
        .await;
        assert_eq!(
            String::from_utf8(crate::console::read(&sandbox, PATH).await.unwrap()).unwrap(),
            "new"
        );
    }

    #[tokio::test]
    async fn test_write_validation_errors() {
        let msg = call(&test_sandbox().await, to_value!({ "content": "x" })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }
}
