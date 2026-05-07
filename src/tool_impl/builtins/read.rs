use std::path::Path;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_CHARS: usize = 2000;

const DESCRIPTION: &str = "Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- When you already know which part of the file you need, only read that part. This can be important for larger files.
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 2000 characters are truncated.";

pub async fn build_read_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("read")
        .description(DESCRIPTION)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once"
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once."
                }
            },
            "required": ["file_path"]
        }))
        .build();

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let Some(path) = args.pointer("/file_path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: file_path",
                "phase": "validation"
            });
        };
        let offset = args
            .pointer("/offset")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1) as usize)
            .unwrap_or(1);
        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(0) as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let bytes = match ctx.runenv.read(Path::new(path)).await {
            Ok(b) => b,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("read {path}: {e}"),
                    "phase": "io"
                });
            }
        };
        let text = String::from_utf8_lossy(&bytes);
        let total: usize = text.lines().count();
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
        crate::to_value!({
            "content": out,
            "total_lines": total as i64
        })
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
    async fn test_read_returns_numbered_lines() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "alpha\nbeta\ngamma\n").unwrap();
        let tool = build_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({ "file_path": tmp.path().to_string_lossy().to_string() }),
                local_ctx(),
            )
            .await;
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
        assert!(content.lines().next().unwrap().trim_start().starts_with("1\t"));
    }

    #[tokio::test]
    async fn test_read_offset_and_limit() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\nb\nc\nd\ne\n").unwrap();
        let tool = build_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({ "file_path": tmp.path().to_string_lossy().to_string(), "offset": 2, "limit": 2 }),
                local_ctx(),
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
        let tool = build_read_tool().await.unwrap().make(&spec());
        let msg = tool.call_next(to_value!({}), local_ctx()).await;
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
        let tool = build_read_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(
                to_value!({ "file_path": "/this/path/does/not/exist/xyz" }),
                local_ctx(),
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
}
