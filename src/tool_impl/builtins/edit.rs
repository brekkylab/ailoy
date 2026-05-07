use std::path::Path;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

const DESCRIPTION: &str = "Performs exact string replacements in files.

Usage:
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.";

pub async fn build_edit_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("edit")
        .description(DESCRIPTION)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify"
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
            "required": ["file_path", "old_string", "new_string"]
        }))
        .build();

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let Some(path) = args.pointer("/file_path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: file_path",
                "phase": "validation"
            });
        };
        let Some(old) = args.pointer("/old_string").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: old_string",
                "phase": "validation"
            });
        };
        let Some(new) = args.pointer("/new_string").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: new_string",
                "phase": "validation"
            });
        };
        if old == new {
            return crate::to_value!({
                "error": "old_string and new_string must differ",
                "phase": "validation"
            });
        }
        let replace_all = args
            .pointer("/replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let bytes = match ctx.runenv.read(Path::new(path)).await {
            Ok(b) => b,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("read {path}: {e}"),
                    "phase": "io"
                });
            }
        };
        let original = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                return crate::to_value!({
                    "error": format!("file {path} is not valid UTF-8"),
                    "phase": "validation"
                });
            }
        };

        let count = original.matches(old).count();
        if count == 0 {
            return crate::to_value!({
                "error": "old_string not found in file",
                "phase": "no_match"
            });
        }
        if !replace_all && count > 1 {
            return crate::to_value!({
                "error": format!(
                    "old_string is not unique ({count} occurrences); pass replace_all=true \
                     or provide more surrounding context"
                ),
                "phase": "ambiguous_match"
            });
        }

        let updated = if replace_all {
            original.replace(old, new)
        } else {
            original.replacen(old, new, 1)
        };
        let replacements = if replace_all { count } else { 1 };

        match ctx
            .runenv
            .write(Path::new(path), updated.as_bytes())
            .await
        {
            Ok(()) => crate::to_value!({
                "ok": true,
                "replacements": replacements as i64
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
    async fn test_edit_unique_replacement() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world\n").unwrap();
        let tool = build_edit_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": tmp.path().to_string_lossy().to_string(),
                    "old_string": "world",
                    "new_string": "ailoy",
                }),
                local_ctx(),
            )
            .await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        );
        assert_eq!(std::fs::read_to_string(tmp.path()).unwrap(), "hello ailoy\n");
    }

    #[tokio::test]
    async fn test_edit_ambiguous_match_fails() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "x\nx\n").unwrap();
        let tool = build_edit_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": tmp.path().to_string_lossy().to_string(),
                    "old_string": "x",
                    "new_string": "y",
                }),
                local_ctx(),
            )
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
        let tool = build_edit_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": tmp.path().to_string_lossy().to_string(),
                    "old_string": "x",
                    "new_string": "y",
                    "replace_all": true,
                }),
                local_ctx(),
            )
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
        let tool = build_edit_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": tmp.path().to_string_lossy().to_string(),
                    "old_string": "zzz",
                    "new_string": "yyy",
                }),
                local_ctx(),
            )
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
        let tool = build_edit_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({
                    "file_path": "/tmp/anything",
                    "old_string": "same",
                    "new_string": "same",
                }),
                local_ctx(),
            )
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
