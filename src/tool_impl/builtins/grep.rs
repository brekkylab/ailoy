use std::path::{Path, PathBuf};

use fancy_regex::{Regex, RegexBuilder};

use crate::{
    runenv::Dirent,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 1000;
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Convert a basename glob (`*.rs`, `test_*`) into an anchored regex.
fn glob_to_regex(glob: &str) -> Result<Regex, String> {
    let mut pat = String::from("^");
    let mut chars = glob.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => pat.push_str(".*"),
            '?' => pat.push('.'),
            '[' => {
                pat.push('[');
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    pat.push(nc);
                    if nc == ']' {
                        break;
                    }
                }
            }
            '.' | '+' | '(' | ')' | '{' | '}' | '|' | '^' | '$' | '\\' => {
                pat.push('\\');
                pat.push(c);
            }
            _ => pat.push(c),
        }
    }
    pat.push('$');
    Regex::new(&pat).map_err(|e| format!("invalid glob {glob:?}: {e}"))
}

/// Treat a file as binary if its first chunk contains a NUL byte (matches what
/// GNU grep does by default with `-I`).
fn is_binary(bytes: &[u8]) -> bool {
    let probe_len = bytes.len().min(8192);
    bytes[..probe_len].contains(&0)
}

fn collect_files(
    root: &Path,
    entries: &[Dirent],
    include_re: Option<&Regex>,
    out: &mut Vec<PathBuf>,
) {
    for entry in entries {
        let entry_path = root.join(entry.name());
        match entry {
            Dirent::File { name, .. } => {
                let matches = include_re
                    .map(|re| re.is_match(name).unwrap_or(false))
                    .unwrap_or(true);
                if matches {
                    out.push(entry_path);
                }
            }
            Dirent::Dir { children, .. } => {
                collect_files(&entry_path, children, include_re, out);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Content,
    FilesWithMatches,
    Count,
}

struct LineHit {
    path: String,
    line_no: usize,
    line: String,
}

fn search_file(
    pattern: &Regex,
    path: &Path,
    bytes: &[u8],
    context_before: usize,
    context_after: usize,
    hits: &mut Vec<LineHit>,
    file_match_count: &mut usize,
    line_limit: usize,
) -> bool {
    if is_binary(bytes) {
        return false;
    }
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut matched_idx: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if pattern.is_match(line).unwrap_or(false) {
            matched_idx.push(i);
        }
    }
    if matched_idx.is_empty() {
        return false;
    }
    *file_match_count += matched_idx.len();
    let path_str = path.to_string_lossy().into_owned();
    let mut emitted: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for &i in &matched_idx {
        let start = i.saturating_sub(context_before);
        let end = (i + context_after).min(lines.len().saturating_sub(1));
        for j in start..=end {
            if emitted.insert(j) && hits.len() < line_limit {
                hits.push(LineHit {
                    path: path_str.clone(),
                    line_no: j + 1,
                    line: lines[j].to_string(),
                });
            }
        }
    }
    true
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
                    "description": "Glob pattern matched against file basenames (e.g. '*.rs'). Applies only when path is a directory."
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
                    "description": "Maximum number of result entries (lines or files) to return (default 1000).",
                    "minimum": 1,
                    "default": 1000
                }
            },
            "required": ["pattern", "path"]
        }))
        .build()
}

pub fn get_grep_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, runenv: &dyn RunEnv| -> Value {
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

        let pattern = match RegexBuilder::new(pattern_str)
            .case_insensitive(case_insensitive)
            .build()
        {
            Ok(re) => re,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("invalid regex {pattern_str:?}: {e}"),
                    "phase": "validation",
                });
            }
        };

        let include_re = match args.pointer("/include").and_then(|v| v.as_str()) {
            Some(g) => match glob_to_regex(g) {
                Ok(re) => Some(re),
                Err(e) => {
                    return crate::to_value!({
                        "error": e,
                        "phase": "validation",
                    });
                }
            },
            None => None,
        };

        let output_mode = match args
            .pointer("/output_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("content")
        {
            "content" => OutputMode::Content,
            "files_with_matches" => OutputMode::FilesWithMatches,
            "count" => OutputMode::Count,
            other => {
                return crate::to_value!({
                    "error": format!("invalid output_mode {other:?}"),
                    "phase": "validation",
                });
            }
        };

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

        // Build the candidate file list.
        let candidate_files: Vec<PathBuf> = match runenv.ls(path).await {
            Ok(entries) => {
                let mut files = Vec::new();
                collect_files(
                    &PathBuf::from(path_str),
                    &entries,
                    include_re.as_ref(),
                    &mut files,
                );
                files
            }
            // ls failed — maybe `path` is a regular file. Fall back to a single-file search.
            Err(_) => vec![PathBuf::from(path_str)],
        };

        let mut hits: Vec<LineHit> = Vec::new();
        let mut files_matched: Vec<String> = Vec::new();
        let mut total_match_count: usize = 0;
        let mut truncated = false;

        for file in &candidate_files {
            let bytes = match runenv.read(file).await {
                Ok(b) => b,
                Err(_) => continue,
            };
            if bytes.len() > MAX_FILE_BYTES {
                continue;
            }
            let mut file_count: usize = 0;
            let line_capacity = if output_mode == OutputMode::Content {
                limit
            } else {
                usize::MAX
            };
            let any = search_file(
                &pattern,
                file,
                &bytes,
                context_before,
                context_after,
                &mut hits,
                &mut file_count,
                line_capacity,
            );
            if any {
                files_matched.push(file.to_string_lossy().into_owned());
                total_match_count += file_count;
            }

            match output_mode {
                OutputMode::Content => {
                    if hits.len() >= limit {
                        truncated = true;
                        break;
                    }
                }
                OutputMode::FilesWithMatches => {
                    if files_matched.len() >= limit {
                        truncated = true;
                        break;
                    }
                }
                OutputMode::Count => {}
            }
        }

        match output_mode {
            OutputMode::Content => {
                let matches = hits
                    .into_iter()
                    .map(|h| {
                        crate::to_value!({
                            "path": h.path.as_str(),
                            "line_no": h.line_no as i64,
                            "line": h.line.as_str(),
                        })
                    })
                    .collect::<Vec<_>>();
                let count = matches.len() as i64;
                crate::to_value!({
                    "matches": matches,
                    "count": count,
                    "truncated": truncated,
                })
            }
            OutputMode::FilesWithMatches => {
                let count = files_matched.len() as i64;
                let files_v = crate::datatype::Value::array(files_matched);
                crate::to_value!({
                    "files": files_v,
                    "count": count,
                    "truncated": truncated,
                })
            }
            OutputMode::Count => {
                crate::to_value!({
                    "match_count": total_match_count as i64,
                    "files_matched": files_matched.len() as i64,
                })
            }
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
        p.insert_func("grep", get_grep_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_grep_tool_desc()]).unwrap();
        let f = funcs.get("grep").unwrap();
        f.call(args, "1", &Local {}).next().await.unwrap().message
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
        let val = msg.contents[0].as_value().unwrap();
        let matches = val.pointer("/matches").and_then(|v| v.as_array()).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| {
            m.pointer("/path")
                .and_then(|v| v.as_str())
                .unwrap()
                .ends_with("a.txt")
        }));
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
        let val = msg.contents[0].as_value().unwrap();
        let matches = val.pointer("/matches").and_then(|v| v.as_array()).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0]
                .pointer("/line_no")
                .and_then(|v| v.as_integer())
                .unwrap(),
            2
        );
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
        let val = msg.contents[0].as_value().unwrap();
        let matches = val.pointer("/matches").and_then(|v| v.as_array()).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0]
                .pointer("/path")
                .and_then(|v| v.as_str())
                .unwrap()
                .ends_with("a.rs")
        );
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
        let val = msg.contents[0].as_value().unwrap();
        let files = val.pointer("/files").and_then(|v| v.as_array()).unwrap();
        assert_eq!(files.len(), 2);
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
        let val = msg.contents[0].as_value().unwrap();
        assert_eq!(
            val.pointer("/match_count")
                .and_then(|v| v.as_integer())
                .unwrap(),
            4
        );
        assert_eq!(
            val.pointer("/files_matched")
                .and_then(|v| v.as_integer())
                .unwrap(),
            2
        );
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
        let val = msg.contents[0].as_value().unwrap();
        let matches = val.pointer("/matches").and_then(|v| v.as_array()).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0]
                .pointer("/path")
                .and_then(|v| v.as_str())
                .unwrap()
                .ends_with("text.txt")
        );
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
        let val = msg.contents[0].as_value().unwrap();
        let matches = val.pointer("/matches").and_then(|v| v.as_array()).unwrap();
        let line_nos: Vec<i64> = matches
            .iter()
            .map(|m| m.pointer("/line_no").and_then(|v| v.as_integer()).unwrap())
            .collect();
        assert_eq!(line_nos, vec![2, 3, 4]);
    }

    #[tokio::test]
    async fn test_grep_invalid_regex() {
        let msg = call(to_value!({
            "pattern": "(",
            "path": "/tmp",
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
