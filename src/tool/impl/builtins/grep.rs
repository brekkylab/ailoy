use std::path::Path;

use crate::{
    runenv::Console,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
    util::truncate::middle_truncate,
};

const DEFAULT_LIMIT: usize = 1000;
const MAX_OUTPUT_CHARS: usize = 200_000;

/// POSIX-safe shell quoting: wrap `s` in single quotes, escaping embedded `'`.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Probe for ripgrep by asking the shell to locate it. Exit 0 means available.
async fn has_ripgrep(console: &dyn Console) -> bool {
    console
        .exec_shell("command -v rg >/dev/null 2>&1".to_string(), Some(5))
        .await
        .map(|r| r.exit_code == 0)
        .unwrap_or(false)
}

pub fn get_grep_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("grep")
        .description(concat!(
            "Search file contents for a regex pattern (Rust/PCRE-style). ",
            "If `path` is a directory, descends recursively. ",
            "Skips files that look binary (contain NUL bytes in their first 8 KiB). ",
            "Files larger than 10 MiB are skipped. ",
            "Use `include` to filter by basename glob (e.g., '*.rs'). ",
            "`output_mode` selects between 'content' (line hits), 'files_with_matches', or 'count'. ",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to match within each line"
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file or directory to search"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern matched against file paths (e.g. '*.rs'). Applies only when path is a directory."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Match case-insensitively (default false)",
                    "default": false
                },
                "output_mode": {
                    "type": "string",
                    "description": "What to return: 'content' (matching lines, default), 'files_with_matches', or 'count'.",
                    "enum": ["content", "files_with_matches", "count"],
                    "default": "content"
                },
                "context_before": {
                    "type": "integer",
                    "description": "Lines of context to include before each match (output_mode='content' only).",
                    "minimum": 0,
                    "default": 0
                },
                "context_after": {
                    "type": "integer",
                    "description": "Lines of context to include after each match (output_mode='content' only).",
                    "minimum": 0,
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of output lines to return (default 1000).",
                    "minimum": 1,
                    "default": 1000
                }
            },
            "required": ["pattern", "path"]
        }))
        .build()
}

pub fn get_grep_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, console: &dyn Console| -> Value {
        let Some(pattern_str) = args.pointer("/pattern").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: pattern",
                "phase": "validation",
            });
        };
        let Some(path_str) = args.pointer("/path").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: path",
                "phase": "validation",
            });
        };
        let path = Path::new(path_str);
        if !path.is_absolute() {
            return crate::to_value!({
                "error": "path must be absolute",
                "phase": "validation",
            });
        }

        let case_insensitive = args
            .pointer("/case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include = args
            .pointer("/include")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let output_mode = args
            .pointer("/output_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("content")
            .to_string();
        let context_before = args
            .pointer("/context_before")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(0) as usize)
            .unwrap_or(0);
        let context_after = args
            .pointer("/context_after")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(0) as usize)
            .unwrap_or(0);
        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1) as usize)
            .unwrap_or(DEFAULT_LIMIT);

        if !matches!(
            output_mode.as_str(),
            "content" | "files_with_matches" | "count"
        ) {
            return crate::to_value!({
                "error": format!("invalid output_mode {output_mode:?}"),
                "phase": "validation",
            });
        }

        let os = console.get_os();
        if os != "linux" && os != "macos" {
            return crate::to_value!({
                "error": format!("grep: unsupported OS '{os}'"),
                "phase": "validation",
            });
        }

        let use_rg = has_ripgrep(console).await;

        // Build a per-tool arg list. The shared shape is roughly:
        //   <flags> [-B N] [-A N] [include-glob] -e PATTERN -- PATH
        // rg differs from grep in: no `-r` (always recursive on dirs), `-g GLOB`
        // instead of `--include=GLOB`, and `--no-heading` to get inline output.
        let (program, tool_args, tool_name): (&str, Vec<String>, &'static str) = if use_rg {
            let mut a = vec![
                "--color=never".to_string(),
                "--no-heading".to_string(),
                "-n".to_string(),
                "-H".to_string(),
            ];
            if case_insensitive {
                a.push("-i".to_string());
            }
            match output_mode.as_str() {
                "files_with_matches" => a.push("-l".to_string()),
                "count" => a.push("-c".to_string()),
                _ => {
                    if context_before > 0 {
                        a.push("-B".to_string());
                        a.push(context_before.to_string());
                    }
                    if context_after > 0 {
                        a.push("-A".to_string());
                        a.push(context_after.to_string());
                    }
                }
            }
            if let Some(g) = &include {
                a.push("-g".to_string());
                a.push(g.clone());
            }
            a.push("-e".to_string());
            a.push(pattern_str.to_string());
            a.push("--".to_string());
            a.push(path_str.to_string());
            ("rg", a, "rg")
        } else {
            let mut a = vec![
                "-r".to_string(),
                "-n".to_string(),
                "-H".to_string(),
                "-E".to_string(),
                "-I".to_string(),
                "--color=never".to_string(),
            ];
            if case_insensitive {
                a.push("-i".to_string());
            }
            match output_mode.as_str() {
                "files_with_matches" => a.push("-l".to_string()),
                "count" => a.push("-c".to_string()),
                _ => {
                    if context_before > 0 {
                        a.push("-B".to_string());
                        a.push(context_before.to_string());
                    }
                    if context_after > 0 {
                        a.push("-A".to_string());
                        a.push(context_after.to_string());
                    }
                }
            }
            if let Some(g) = &include {
                a.push(format!("--include={g}"));
            }
            a.push("-e".to_string());
            a.push(pattern_str.to_string());
            a.push("--".to_string());
            a.push(path_str.to_string());
            ("grep", a, "grep")
        };

        // Quote every argument for embedding in a single `sh -c` command line.
        // Flags are literal-safe but uniform quoting keeps the assembly simple.
        let mut cmd = String::from(program);
        for arg in &tool_args {
            cmd.push(' ');
            cmd.push_str(&sh_quote(arg));
        }

        let result = match console.exec_shell(cmd, None).await {
            Ok(r) => r,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("exec failed: {e}"),
                    "phase": "io",
                });
            }
        };

        // 0 = matches, 1 = no matches, 2+ = real error (invalid regex, missing
        // path, etc.). Spawn failure surfaces as exit_code = -1.
        if result.exit_code >= 2 || result.exit_code < 0 {
            return crate::to_value!({
                "error": format!(
                    "{tool_name} failed (exit {}): {}",
                    result.exit_code,
                    result.stderr.trim()
                ),
                "phase": "io",
            });
        }

        let lines: Vec<&str> = result.stdout.lines().collect();
        let total = lines.len();
        let truncated = total > limit;
        let kept = if truncated {
            &lines[..limit]
        } else {
            &lines[..]
        };
        let output_text = kept.join("\n");
        let output_text = middle_truncate(output_text, MAX_OUTPUT_CHARS);

        crate::to_value!({
            "output": output_text.as_str(),
            "tool": tool_name,
            "count": kept.len() as i64,
            "truncated": truncated,
        })
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        datatype::Value,
        message::Message,
        runenv::LocalConsole,
        to_value,
        tool::ToolProvider,
    };

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("grep", get_grep_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_grep_tool_desc()]).unwrap();
        let f = funcs.get("grep").unwrap();
        let local = LocalConsole::new();
        let console = &local;
        f.call(args, "1", console).next().await.unwrap().message
    }

    fn output(msg: &Message) -> String {
        msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/output")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn test_grep_finds_matches_in_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.txt"), "hello world\nbye world\n").unwrap();
        std::fs::write(root.join("b.txt"), "nothing\n").unwrap();

        let msg = call(to_value!({
            "pattern": "world",
            "path": root.to_string_lossy().to_string(),
        }))
        .await;
        let out = output(&msg);
        let world_lines: Vec<&str> = out.lines().filter(|l| l.contains("world")).collect();
        assert_eq!(world_lines.len(), 2);
        assert!(world_lines.iter().all(|l| l.contains("a.txt")));
    }

    #[tokio::test]
    async fn test_grep_single_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "alpha\nbeta\ngamma\n").unwrap();
        let msg = call(to_value!({
            "pattern": "^beta$",
            "path": tmp.path().to_string_lossy().to_string(),
        }))
        .await;
        let out = output(&msg);
        assert_eq!(out.lines().count(), 1);
        assert!(out.contains("beta"));
        // both rg and grep emit `path:lineno:text` — line 2 should appear.
        assert!(out.contains(":2:"));
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "Hello\nGOODBYE\n").unwrap();
        let msg = call(to_value!({
            "pattern": "hello",
            "path": tmp.path().to_string_lossy().to_string(),
            "case_insensitive": true,
        }))
        .await;
        let val = msg.contents[0].as_value().unwrap();
        assert_eq!(
            val.pointer("/count").and_then(|v| v.as_integer()).unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn test_grep_include_filter() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "needle\n").unwrap();
        std::fs::write(root.join("b.txt"), "needle\n").unwrap();

        let msg = call(to_value!({
            "pattern": "needle",
            "path": root.to_string_lossy().to_string(),
            "include": "*.rs",
        }))
        .await;
        let out = output(&msg);
        assert_eq!(out.lines().count(), 1);
        assert!(out.contains("a.rs"));
        assert!(!out.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_grep_files_with_matches_mode() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.txt"), "needle\n").unwrap();
        std::fs::write(root.join("b.txt"), "needle\nneedle\n").unwrap();
        std::fs::write(root.join("c.txt"), "nope\n").unwrap();

        let msg = call(to_value!({
            "pattern": "needle",
            "path": root.to_string_lossy().to_string(),
            "output_mode": "files_with_matches",
        }))
        .await;
        let out = output(&msg);
        let files: Vec<&str> = out.lines().collect();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.ends_with("a.txt")));
        assert!(files.iter().any(|f| f.ends_with("b.txt")));
    }

    #[tokio::test]
    async fn test_grep_count_mode() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.txt"), "x\nx\nx\n").unwrap();
        std::fs::write(root.join("b.txt"), "x\n").unwrap();

        let msg = call(to_value!({
            "pattern": "x",
            "path": root.to_string_lossy().to_string(),
            "output_mode": "count",
        }))
        .await;
        let out = output(&msg);
        // Output is `path:N` per file. Sum the Ns.
        let total: i64 = out
            .lines()
            .filter_map(|l| l.rsplit(':').next().and_then(|n| n.parse::<i64>().ok()))
            .sum();
        assert_eq!(total, 4);
    }

    #[tokio::test]
    async fn test_grep_skips_binary() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut bin = b"hello".to_vec();
        bin.push(0);
        bin.extend_from_slice(b"needle");
        std::fs::write(root.join("bin"), bin).unwrap();
        std::fs::write(root.join("text.txt"), "needle\n").unwrap();

        let msg = call(to_value!({
            "pattern": "needle",
            "path": root.to_string_lossy().to_string(),
        }))
        .await;
        let out = output(&msg);
        assert!(out.contains("text.txt"));
        // The binary should not show up as a content line. `grep -I` and rg both
        // skip it; without `-I` grep would emit a "Binary file … matches" line.
        assert!(!out.lines().any(|l| l.contains("/bin:")));
    }

    #[tokio::test]
    async fn test_grep_context_lines() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\nb\nMATCH\nc\nd\n").unwrap();
        let msg = call(to_value!({
            "pattern": "MATCH",
            "path": tmp.path().to_string_lossy().to_string(),
            "context_before": 1,
            "context_after": 1,
        }))
        .await;
        let out = output(&msg);
        // Both rg and grep print 3 lines: context (`-`), match (`:`), context (`-`).
        assert_eq!(out.lines().count(), 3);
        assert!(out.contains("MATCH"));
        assert!(out.contains("b"));
        assert!(out.contains("c"));
    }

    #[tokio::test]
    async fn test_grep_relative_path_rejected() {
        let msg = call(to_value!({
            "pattern": "x",
            "path": "rel/path",
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
