use std::path::{Path, PathBuf};

use fancy_regex::Regex;

use crate::{
    runenv::Dirent,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 1000;

/// Convert a shell-style glob (matched against a path basename) into a regex
/// anchored at both ends. Supports `*`, `?`, and character classes via `[..]`.
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

/// Walk the dirent tree rooted at `root_path`, collecting paths that satisfy
/// `keep`. Stops once `limit` matches have been gathered.
fn walk(
    root_path: &Path,
    entries: &[Dirent],
    type_filter: Option<char>,
    name_re: Option<&Regex>,
    max_depth: Option<usize>,
    depth: usize,
    out: &mut Vec<String>,
    limit: usize,
) {
    for entry in entries {
        if out.len() >= limit {
            return;
        }
        let entry_path = root_path.join(entry.name());
        let matches_type = match type_filter {
            Some('f') => entry.is_file(),
            Some('d') => entry.is_dir(),
            _ => true,
        };
        let matches_name = name_re
            .map(|re| re.is_match(entry.name()).unwrap_or(false))
            .unwrap_or(true);
        if matches_type && matches_name {
            out.push(entry_path.to_string_lossy().into_owned());
        }
        if let Some(children) = entry.children() {
            let next_depth = depth + 1;
            if max_depth.is_none_or(|m| next_depth <= m) {
                walk(
                    &entry_path,
                    children,
                    type_filter,
                    name_re,
                    max_depth,
                    next_depth,
                    out,
                    limit,
                );
            }
        }
    }
}

pub fn get_find_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("find")
        .description(concat!(
            "Recursively find files and directories under a base path. ",
            "Filters by glob pattern on the basename and/or by entry type. ",
            "Returns paths sorted by depth then name. ",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute base directory to search in"
                },
                "name": {
                    "type": "string",
                    "description": "Glob pattern matched against the basename (e.g. '*.rs', 'test_*'). Matches all entries when omitted."
                },
                "type": {
                    "type": "string",
                    "description": "Restrict results to 'f' (regular files) or 'd' (directories). Matches both when omitted.",
                    "enum": ["f", "d"]
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursion depth (1 = direct children only). Unlimited when omitted.",
                    "minimum": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of paths to return (default 1000).",
                    "minimum": 1,
                    "default": 1000
                }
            },
            "required": ["path"]
        }))
        .build()
}

pub fn get_find_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, runenv: &dyn RunEnv| -> Value {
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

        let name_re = match args.pointer("/name").and_then(|v| v.as_str()) {
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

        let type_filter = match args.pointer("/type").and_then(|v| v.as_str()) {
            Some("f") => Some('f'),
            Some("d") => Some('d'),
            Some(other) => {
                return crate::to_value!({
                    "error": format!("invalid type {other:?}; expected 'f' or 'd'"),
                    "phase": "validation",
                });
            }
            None => None,
        };

        let max_depth = args
            .pointer("/max_depth")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1) as usize);

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
            &entries,
            type_filter,
            name_re.as_ref(),
            max_depth,
            1,
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
        p.insert_func("find", get_find_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_find_tool_desc()]).unwrap();
        let f = funcs.get("find").unwrap();
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
    async fn test_find_lists_recursively() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.rs"), "").unwrap();
        std::fs::write(root.join("sub/c.txt"), "").unwrap();

        let msg = call(to_value!({ "path": root.to_string_lossy().to_string() })).await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("a.rs")));
        assert!(p.iter().any(|x| x.ends_with("sub/b.rs")));
        assert!(p.iter().any(|x| x.ends_with("sub/c.txt")));
        assert!(p.iter().any(|x| x.ends_with("sub")));
    }

    #[tokio::test]
    async fn test_find_name_filter() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "").unwrap();
        std::fs::write(root.join("b.txt"), "").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/c.rs"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "name": "*.rs",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("a.rs")));
        assert!(p.iter().any(|x| x.ends_with("c.rs")));
        assert!(!p.iter().any(|x| x.ends_with("b.txt")));
    }

    #[tokio::test]
    async fn test_find_type_filter() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a"), "").unwrap();
        std::fs::create_dir(root.join("d")).unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "type": "d",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/d")));
        assert!(!p.iter().any(|x| x.ends_with("/a")));
    }

    #[tokio::test]
    async fn test_find_max_depth() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("top"), "").unwrap();
        std::fs::write(root.join("a/mid"), "").unwrap();
        std::fs::write(root.join("a/b/deep"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "max_depth": 1,
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/top")));
        assert!(p.iter().any(|x| x.ends_with("/a")));
        assert!(!p.iter().any(|x| x.ends_with("/mid")));
        assert!(!p.iter().any(|x| x.ends_with("/deep")));
    }

    #[tokio::test]
    async fn test_find_limit_marks_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for i in 0..5 {
            std::fs::write(root.join(format!("f{i}")), "").unwrap();
        }
        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "limit": 2,
        }))
        .await;
        let val = msg.contents[0].as_value().unwrap();
        assert_eq!(val.pointer("/count").and_then(|v| v.as_integer()).unwrap(), 2);
        assert!(val.pointer("/truncated").and_then(|v| v.as_bool()).unwrap());
    }

    #[tokio::test]
    async fn test_find_relative_path_rejected() {
        let msg = call(to_value!({ "path": "rel/path" })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }
}
