use std::path::Path;

use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_LIMIT: usize = 1000;

/// POSIX single-quote escaping for embedding `s` between `'…'` in a script.
fn sh_single_quote_inner(s: &str) -> String {
    s.replace('\'', "'\\''")
}

/// Convert a glob pattern to a POSIX extended regular expression.
///   `**/`   → `(.*/)?` (zero or more complete path segments; anchored to a
///                       separator so `**/foo` matches `foo` and `a/foo`, but
///                       not `barfoo`)
///   `**`    → `.*`     (when not followed by `/`, e.g. trailing `src/**`)
///   `*`     → `[^/]*`  (within a single segment)
///   `?`     → `[^/]`
///   `[..]`  → `[..]`   (character class, with `!` → `^` negation)
///   regex metacharacters elsewhere (`.`, `+`, `(`, `)`, `|`, `^`, `$`, `\`,
///   `{`, `}`) are backslash-escaped so they match literally.
fn glob_to_ere(pat: &str) -> Result<String, String> {
    let mut out = String::with_capacity(pat.len());
    let chars: Vec<char> = pat.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '*' {
            if i + 1 < chars.len() && chars[i + 1] == '*' {
                if i + 2 < chars.len() && chars[i + 2] == '/' {
                    out.push_str("(.*/)?");
                    i += 3;
                } else {
                    out.push_str(".*");
                    i += 2;
                }
            } else {
                out.push_str("[^/]*");
                i += 1;
            }
        } else if c == '?' {
            out.push_str("[^/]");
            i += 1;
        } else if c == '[' {
            let start = i;
            out.push('[');
            i += 1;
            if i < chars.len() && (chars[i] == '!' || chars[i] == '^') {
                out.push('^');
                i += 1;
            }
            while i < chars.len() && chars[i] != ']' {
                out.push(chars[i]);
                i += 1;
            }
            if i >= chars.len() {
                return Err(format!(
                    "unterminated character class starting at position {start}"
                ));
            }
            out.push(']');
            i += 1;
        } else if matches!(
            c,
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '\\' | '{' | '}'
        ) {
            out.push('\\');
            out.push(c);
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
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

        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1) as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let os = console.get_os();
        if os != "linux" && os != "macos" {
            return crate::to_value!({
                "error": format!("glob: unsupported OS '{os}'"),
                "phase": "validation",
            });
        }

        let regex = match glob_to_ere(pattern_str) {
            Ok(r) => r,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("invalid pattern: {e}"),
                    "phase": "validation",
                });
            }
        };
        let base_q = sh_single_quote_inner(path_str);
        let regex_q = sh_single_quote_inner(&regex);

        // Stay in POSIX `sh` (via `exec_shell`) so this works on any sandbox
        // image, including ones without zsh or bash 4+. `find` walks the tree,
        // `grep -E` filters via a glob-to-ERE conversion (handles `**`), and a
        // shell read-loop prefixes the absolute base path.
        let script = format!(
            r#"base='{base_q}'
cd "$base" || exit 1
find . -type f 2>/dev/null | grep -E '^\./{regex_q}$' | while IFS= read -r f; do
  printf '%s%s\n' "$base" "${{f#.}}"
done"#,
        );

        let result = match console.exec_shell(script, None).await {
            Ok(r) => r,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("exec failed: {e}"),
                    "phase": "io",
                });
            }
        };
        if result.exit_code != 0 {
            return crate::to_value!({
                "error": format!("glob failed (exit {}): {}", result.exit_code, result.stderr),
                "phase": "io",
            });
        }

        let mut paths: Vec<String> = result
            .stdout
            .lines()
            .take(limit + 1)
            .map(|s| s.to_string())
            .collect();
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
    use crate::{
        datatype::Value,
        message::Message,
        runenv::LocalConsole,
        to_value,
        tool::ToolProvider,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Simulate the exact grep pattern used in the glob script:
    ///   grep -E '^\./<ere>$'
    /// and test it against a find(1)-style path like "./dir/file".
    fn ere_matches_find_path(ere: &str, find_path: &str) -> bool {
        let pattern = format!("^\\./{ere}$");
        fancy_regex::Regex::new(&pattern)
            .expect("glob_to_ere produced invalid ERE")
            .is_match(find_path)
            .unwrap()
    }

    // ── unit tests: glob_to_ere ───────────────────────────────────────────────

    /// `*` must stay within one path segment — `[^/]*`, not `.*`.
    #[test]
    fn glob_to_ere_single_star_does_not_cross_slash() {
        assert_eq!(glob_to_ere("*").unwrap(), "[^/]*");
    }

    /// `?` must produce `[^/]` so it never matches a path separator.
    #[test]
    fn glob_to_ere_question_mark_is_single_non_slash() {
        let ere = glob_to_ere("fo?").unwrap();
        assert!(ere_matches_find_path(&ere, "./foo"), "should match foo");
        assert!(!ere_matches_find_path(&ere, "./fo/"), "? must not match /");
        assert!(
            !ere_matches_find_path(&ere, "./fooo"),
            "? is exactly one char"
        );
    }

    /// `.` in a glob pattern is a literal dot, not an ERE wildcard.
    #[test]
    fn glob_to_ere_dot_is_escaped_to_literal() {
        let ere = glob_to_ere("foo.rs").unwrap();
        assert!(ere_matches_find_path(&ere, "./foo.rs"));
        assert!(
            !ere_matches_find_path(&ere, "./fooXrs"),
            "unescaped dot would match any char — escaping is broken"
        );
    }

    /// ERE metacharacters other than `.` must also be escaped.
    #[test]
    fn glob_to_ere_regex_metachars_are_escaped() {
        for meta in ['+', '(', ')', '|', '^', '$', '{', '}'] {
            let pat = format!("foo{meta}bar");
            let ere = glob_to_ere(&pat).unwrap();
            assert!(
                ere_matches_find_path(&ere, &format!("./foo{meta}bar")),
                "literal '{meta}' should match itself"
            );
            assert!(
                !ere_matches_find_path(&ere, "./fooXbar"),
                "'{meta}' was not escaped — it acted as a regex operator"
            );
        }
    }

    /// `[!abc]` glob negation must become `[^abc]` in the ERE.
    #[test]
    fn glob_to_ere_char_class_negation_converts_bang_to_caret() {
        let ere = glob_to_ere("[!abc].rs").unwrap();
        assert!(
            ere.contains("[^abc]"),
            "expected [^abc] in ERE, got: {ere:?}"
        );
        assert!(ere_matches_find_path(&ere, "./d.rs"), "d is not in [abc]");
        assert!(!ere_matches_find_path(&ere, "./a.rs"), "a is in [abc]");
    }

    /// `**/foo.rs` must match at root and any depth, but must NOT match a
    /// partial filename like `barfoo.rs`. Fixed by translating `**/` → `(.*/)?`
    /// which anchors the match to a segment boundary.
    #[test]
    fn glob_to_ere_double_star_no_partial_filename_false_positive() {
        let ere = glob_to_ere("**/foo.rs").unwrap();
        assert!(
            ere_matches_find_path(&ere, "./foo.rs"),
            "**/ should match at root"
        );
        assert!(
            ere_matches_find_path(&ere, "./src/foo.rs"),
            "**/ should match one level deep"
        );
        assert!(
            ere_matches_find_path(&ere, "./a/b/c/foo.rs"),
            "**/ should match any depth"
        );
        assert!(
            !ere_matches_find_path(&ere, "./barfoo.rs"),
            "** must require a path separator boundary — barfoo.rs must not match **/foo.rs"
        );
    }

    /// `[]abc]` — bracket-first syntax: `]` as the first char in `[...]` is a
    /// literal `]`. Because ERE uses the same rule, the parser output `[]abc]`
    /// happens to be a valid ERE that matches `]`, `a`, `b`, `c` correctly.
    #[test]
    fn glob_to_ere_bracket_first_closing_bracket_in_class() {
        let ere = glob_to_ere("[]abc]").unwrap();
        let re = fancy_regex::Regex::new(&format!("^{ere}$"))
            .expect("bracket-first class produced invalid ERE");
        for ch in [']', 'a', 'b', 'c'] {
            assert!(
                re.is_match(&ch.to_string()).unwrap(),
                "'{ch}' should be matched by []abc]"
            );
        }
        assert!(!re.is_match("d").unwrap(), "'d' must not match []abc]");
    }

    /// An unterminated character class (`[abc` with no `]`) must be rejected
    /// with `Err`, not silently produce an unclosed `[` that makes grep fail.
    #[test]
    fn glob_to_ere_unterminated_char_class_returns_err() {
        assert!(
            glob_to_ere("[abc").is_err(),
            "unterminated '[' must return Err, not a broken ERE string"
        );
    }

    // ── integration tests (end-to-end through a Machine console) ─────────────

    fn provider() -> ToolProvider {
        let mut p = ToolProvider::new();
        p.insert_func("glob", get_glob_tool_func());
        p
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_glob_tool_desc()]).unwrap();
        let f = funcs.get("glob").unwrap();
        let local = LocalConsole::new();
        let console = &local;
        f.call(args, "1", console).next().await.unwrap().message
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
    async fn test_glob_globstar_requires_separator() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("foo.rs"), "").unwrap();
        std::fs::write(root.join("barfoo.rs"), "").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/foo.rs"), "").unwrap();
        std::fs::write(root.join("sub/barfoo.rs"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "**/foo.rs",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/foo.rs")));
        assert!(p.iter().any(|x| x.ends_with("/sub/foo.rs")));
        assert!(!p.iter().any(|x| x.ends_with("barfoo.rs")));
    }

    #[tokio::test]
    async fn test_glob_unterminated_class_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let msg = call(to_value!({
            "path": dir.path().to_string_lossy().to_string(),
            "pattern": "[abc",
        }))
        .await;
        let val = msg.contents[0].as_value().unwrap();
        let phase = val.pointer("/phase").and_then(|v| v.as_str()).unwrap();
        assert_eq!(phase, "validation");
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

    /// `**/foo.rs` must not match a file named `barfoo.rs` in the root.
    #[tokio::test]
    async fn test_glob_double_star_no_partial_filename_false_positive() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // This file shares the suffix "foo.rs" but is NOT named "foo.rs".
        std::fs::write(root.join("barfoo.rs"), "").unwrap();
        // This is the only legitimate match.
        std::fs::create_dir(root.join("src")).unwrap();
        std::fs::write(root.join("src/foo.rs"), "").unwrap();

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "**/foo.rs",
        }))
        .await;
        let p = paths(&msg);
        assert!(
            p.iter().any(|x| x.ends_with("/src/foo.rs")),
            "src/foo.rs should be matched"
        );
        assert!(
            !p.iter().any(|x| x.ends_with("barfoo.rs")),
            "**/foo.rs must not match barfoo.rs (false positive from `**` → `.*`)"
        );
    }

    /// `*.txt` matches a file named `a.txt` but must NOT match `atxt`
    /// (the dot in the pattern is literal, not an ERE `.` wildcard).
    #[tokio::test]
    async fn test_glob_dot_in_extension_is_literal() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.txt"), "").unwrap();
        std::fs::write(root.join("atxt"), "").unwrap(); // no dot

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "*.txt",
        }))
        .await;
        let p = paths(&msg);
        assert!(p.iter().any(|x| x.ends_with("/a.txt")));
        assert!(
            !p.iter().any(|x| x.ends_with("/atxt")),
            "dot in pattern must be literal — atxt must not match *.txt"
        );
    }

    /// `[!rs]*` should match files that do NOT start with `r` or `s`.
    #[tokio::test]
    async fn test_glob_char_class_negation() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("readme.md"), "").unwrap(); // starts with r
        std::fs::write(root.join("setup.sh"), "").unwrap(); // starts with s
        std::fs::write(root.join("main.go"), "").unwrap(); // starts with m → should match

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "[!rs]*",
        }))
        .await;
        let p = paths(&msg);
        assert!(
            p.iter().any(|x| x.ends_with("/main.go")),
            "main.go should match [!rs]*"
        );
        assert!(
            !p.iter().any(|x| x.ends_with("/readme.md")),
            "readme.md starts with r — must not match [!rs]*"
        );
        assert!(
            !p.iter().any(|x| x.ends_with("/setup.sh")),
            "setup.sh starts with s — must not match [!rs]*"
        );
    }

    /// A glob pattern with ERE metacharacters (`+`, `(`) must treat them as
    /// literals, not as regex operators.
    #[tokio::test]
    async fn test_glob_regex_metachars_in_filename_are_literal() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a+b.txt"), "").unwrap();
        std::fs::write(root.join("aab.txt"), "").unwrap(); // would match if + is unescaped

        let msg = call(to_value!({
            "path": root.to_string_lossy().to_string(),
            "pattern": "a+b.txt",
        }))
        .await;
        let p = paths(&msg);
        assert!(
            p.iter().any(|x| x.ends_with("/a+b.txt")),
            "a+b.txt should match the literal pattern a+b.txt"
        );
        assert!(
            !p.iter().any(|x| x.ends_with("/aab.txt")),
            "aab.txt must not match — + is literal in glob, not a quantifier"
        );
    }
}
