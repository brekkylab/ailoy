use std::path::{Path, PathBuf};

use fancy_regex::Regex;

use crate::{
    runenv::Dirent,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 1000;

/// Convert a path glob into a regex anchored at both ends. The glob is matched
/// against a path relative to the search root. Supports:
///
/// * `*`  — any run of characters except `/`
/// * `?`  — a single character except `/`
/// * `**` — any number of path segments (use `**/x` to also match `x` at root)
/// * `[..]` — character classes
pub(super) fn glob_to_regex(glob: &str) -> Result<Regex, String> {
    let mut pat = String::from("^");
    let mut chars = glob.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    if chars.peek() == Some(&'/') {
                        chars.next();
                        pat.push_str("(?:.*/)?");
                    } else {
                        pat.push_str(".*");
                    }
                } else {
                    pat.push_str("[^/]*");
                }
            }
            '?' => pat.push_str("[^/]"),
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

/// Walk the dirent tree rooted at `base`, collecting file paths whose path
/// relative to `base` matches `pat_re`. Stops once `limit` matches are gathered.
fn walk(
    base: &Path,
    rel_prefix: &str,
    entries: &[Dirent],
    pat_re: &Regex,
    out: &mut Vec<String>,
    limit: usize,
) {
    for entry in entries {
        if out.len() >= limit {
            return;
        }
        let rel = if rel_prefix.is_empty() {
            entry.name().to_string()
        } else {
            format!("{}/{}", rel_prefix, entry.name())
        };
        if entry.is_file() && pat_re.is_match(&rel).unwrap_or(false) {
            out.push(base.join(&rel).to_string_lossy().into_owned());
        }
        if let Some(children) = entry.children() {
            walk(base, &rel, children, pat_re, out, limit);
        }
    }
}

pub fn get_glob_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("glob")
        .description(concat!(
            "Fast file pattern matching against a directory tree. ",
            "Returns file paths whose path (relative to `path`) matches `pattern`. ",
            "Supports `*`, `?`, character classes `[..]`, and `**` for any-depth segments ",
            "(e.g. `**/*.rs`, `src/**/*.ts`). ",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern matched against paths relative to `path` (e.g. '**/*.rs', 'src/**/*.ts')."
                },
                "path": {
                    "type": "string",
                    "description": "Absolute base directory to search in"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of paths to return (default 1000).",
                    "minimum": 1,
                    "default": 1000
                }
            },
            "required": ["pattern", "path"]
        }))
        .build()
}

pub fn get_glob_tool_func() -> ToolFunc {
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

        let pat_re = match glob_to_regex(pattern_str) {
            Ok(re) => re,
            Err(e) => {
                return crate::to_value!({
                    "error": e,
                    "phase": "validation",
                });
            }
        };

        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1) as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let entries = match runenv.ls(path).await {
            Ok(e) => e,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("ls {path_str}: {e}"),
                    "phase": "io",
                });
            }
        };

        let mut paths: Vec<String> = Vec::new();
        walk(
            &PathBuf::from(path_str),
            "",
            &entries,
            &pat_re,
            &mut paths,
            limit + 1,
        );
        let truncated = paths.len() > limit;
        if truncated {
            paths.truncate(limit);
        }

        let count = paths.len() as i64;
        let paths_v = crate::datatype::Value::array(paths);
        crate::to_value!({
            "paths": paths_v,
            "count": count,
            "truncated": truncated,
        })
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{datatype::Value, message::Message, runenv::Local, to_value, tool::ToolProvider};

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("glob", get_glob_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_glob_tool_desc()]).unwrap();
        let f = funcs.get("glob").unwrap();
        f.call(args, "1", &Local {}).next().await.unwrap().message
    }

    fn paths(msg: &Message) -> Vec<String> {
        msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/paths")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn test_glob_globstar_matches_any_depth() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.rs"), "").unwrap();
        std::fs::write(root.join("sub/c.txt"), "").unwrap();
        std::fs::create_dir(root.join("sub/deep")).unwrap();
        std::fs::write(root.join("sub/deep/d.rs"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "**/*.rs",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/a.rs")));
        assert!(p.iter().any(|x| x.ends_with("/sub/b.rs")));
        assert!(p.iter().any(|x| x.ends_with("/sub/deep/d.rs")));
        assert!(!p.iter().any(|x| x.ends_with(".txt")));
    }

    #[tokio::test]
    async fn test_glob_single_star_does_not_cross_slashes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.rs"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "*.rs",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/a.rs")));
        assert!(!p.iter().any(|x| x.ends_with("/sub/b.rs")));
    }

    #[tokio::test]
    async fn test_glob_prefixed_globstar() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.ts"), "").unwrap();
        std::fs::create_dir(root.join("src/inner")).unwrap();
        std::fs::write(root.join("src/inner/b.ts"), "").unwrap();
        std::fs::create_dir(root.join("other")).unwrap();
        std::fs::write(root.join("other/c.ts"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "src/**/*.ts",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/src/a.ts")));
        assert!(p.iter().any(|x| x.ends_with("/src/inner/b.ts")));
        assert!(!p.iter().any(|x| x.ends_with("/other/c.ts")));
    }

    #[tokio::test]
    async fn test_glob_returns_files_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a"), "").unwrap();
        std::fs::create_dir(root.join("d")).unwrap();
        std::fs::write(root.join("d/b"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "**/*",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/a")));
        assert!(p.iter().any(|x| x.ends_with("/d/b")));
        assert!(!p.iter().any(|x| x.ends_with("/d")));
    }

    #[tokio::test]
    async fn test_glob_limit_marks_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for i in 0..5 {
            std::fs::write(root.join(format!("f{i}")), "").unwrap();
        }
        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "*",
            "limit": 2,
        }))
        .await;
        let val = msg.contents[0].as_value().unwrap();
        assert_eq!(
            val.pointer("/count").and_then(|v| v.as_integer()).unwrap(),
            2
        );
        assert!(val.pointer("/truncated").and_then(|v| v.as_bool()).unwrap());
    }

    #[tokio::test]
    async fn test_glob_relative_path_rejected() {
        let msg = call(to_value!({ "path": "rel/path", "pattern": "*" })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }
}
