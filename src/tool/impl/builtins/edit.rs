use std::path::Path;

use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

pub fn get_edit_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("edit")
        .description(concat!(
            "Performs exact string replacements in a text file on the local filesystem. ",
            "When matching text from read tool output, preserve indentation exactly as it appears after the line number prefix; never include the line number prefix itself in old_string or new_string. ",
            "The edit fails if old_string is not unique in the file; provide more surrounding context to disambiguate, or pass replace_all=true to change every occurrence. ",
            "Use replace_all to rename a symbol across the file. ",
            "old_string and new_string must differ. ",
            "Binary or non-UTF-8 files return an error.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to modify"
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string (default false)",
                    "default": false
                }
            },
            "required": ["path", "old_string", "new_string"]
        }))
        .build()
}

pub fn get_edit_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, console: &dyn Console| -> Value {
        let Some(path) = args.pointer("/path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: path",
                "phase": "validation",
            });
        };
        let Some(old) = args.pointer("/old_string").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: old_string",
                "phase": "validation",
            });
        };
        let Some(new) = args.pointer("/new_string").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: new_string",
                "phase": "validation",
            });
        };
        if old == new {
            return crate::to_value!({
                "error": "old_string and new_string must differ",
                "phase": "validation",
            });
        }
        let replace_all = args
            .pointer("/replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let bytes = match console.read(Path::new(path)).await {
            Ok(b) => b,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("read {path}: {e}"),
                    "phase": "io",
                });
            }
        };
        let original = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                return crate::to_value!({
                    "error": format!("file {path} is not valid UTF-8"),
                    "phase": "validation",
                });
            }
        };

        let count = original.matches(old).count();
        if count == 0 {
            return crate::to_value!({
                "error": "old_string not found in file",
                "phase": "no_match",
            });
        }
        if !replace_all && count > 1 {
            return crate::to_value!({
                "error": format!(
                    "old_string is not unique ({count} occurrences); pass replace_all=true \
                     or provide more surrounding context"
                ),
                "phase": "ambiguous_match",
            });
        }

        let updated = if replace_all {
            original.replace(old, new)
        } else {
            original.replacen(old, new, 1)
        };
        let replacements = if replace_all { count } else { 1 };

        match console.write(Path::new(path), updated.as_bytes()).await {
            Ok(()) => crate::to_value!({
                "ok": true,
                "replacements": replacements as i64,
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
    use crate::{
        datatype::Value,
        message::Message,
        runenv::{Local, Machine},
        to_value,
        tool::ToolProvider,
    };

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("edit", get_edit_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_edit_tool_desc()]).unwrap();
        let f = funcs.get("edit").unwrap();
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        f.call(args, "1", console).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_edit_unique_replacement() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world\n").unwrap();
        let msg = call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "old_string": "world",
            "new_string": "ailoy",
        }))
        .await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path()).unwrap(),
            "hello ailoy\n"
        );
    }

    #[tokio::test]
    async fn test_edit_ambiguous_match_fails() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "x\nx\n").unwrap();
        let msg = call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "old_string": "x",
            "new_string": "y",
        }))
        .await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "ambiguous_match");
        assert_eq!(std::fs::read_to_string(tmp.path()).unwrap(), "x\nx\n");
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "x\nx\nx\n").unwrap();
        let msg = call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "old_string": "x",
            "new_string": "y",
            "replace_all": true,
        }))
        .await;
        let n = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/replacements")
            .and_then(|v| v.as_integer())
            .unwrap();
        assert_eq!(n, 3);
        assert_eq!(std::fs::read_to_string(tmp.path()).unwrap(), "y\ny\ny\n");
    }

    #[tokio::test]
    async fn test_edit_no_match() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "abc").unwrap();
        let msg = call(to_value!({
            "path": tmp.path().to_string_lossy().to_string(),
            "old_string": "zzz",
            "new_string": "yyy",
        }))
        .await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "no_match");
    }

    #[tokio::test]
    async fn test_edit_old_equals_new_rejected() {
        let msg = call(to_value!({
            "path": "/tmp/anything",
            "old_string": "same",
            "new_string": "same",
        }))
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
