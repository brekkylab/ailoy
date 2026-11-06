use crate::{
    console::Sandbox,
    datatype::Value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

#[derive(Debug)]
enum PatchOp {
    Add {
        path: String,
        content: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
    Delete {
        path: String,
    },
}

#[derive(Debug, Default)]
struct Hunk {
    /// The `@@ <header>` lines before the hunk, in order: each is looked for after
    /// the one before it, and the hunk is matched after the last.
    anchors: Vec<String>,
    before: String,
    after: String,
    /// `*** End of File`: the hunk is at the end of the file.
    at_eof: bool,
}

/// Parse the patch language of Codex's `apply_patch` (see the tool description).
fn parse_patch(text: &str) -> anyhow::Result<Vec<PatchOp>> {
    let trimmed = text.trim_matches('\n');
    let lines: Vec<&str> = trimmed.lines().collect();
    if lines.first().map(|s| s.trim_end()) != Some("*** Begin Patch") {
        anyhow::bail!("patch must start with '*** Begin Patch'");
    }
    if lines.last().map(|s| s.trim_end()) != Some("*** End Patch") {
        anyhow::bail!("patch must end with '*** End Patch'");
    }
    let body = &lines[1..lines.len() - 1];
    // A file operation's header, which ends the lines of the one before it.
    let is_header = |l: &str| l.starts_with("*** ") && l.trim_end() != "*** End of File";

    let mut ops = Vec::new();
    let mut i = 0;
    while i < body.len() {
        let line = body[i];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            i += 1;
            let mut content_lines: Vec<&str> = Vec::new();
            while i < body.len() && !is_header(body[i]) {
                let l = body[i];
                let stripped = l.strip_prefix('+').unwrap_or(l);
                content_lines.push(stripped);
                i += 1;
            }
            let mut content = content_lines.join("\n");
            if !content.is_empty() {
                content.push('\n');
            }
            ops.push(PatchOp::Add {
                path: path.trim().to_string(),
                content,
            });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            i += 1;
            ops.push(PatchOp::Delete {
                path: path.trim().to_string(),
            });
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            i += 1;
            let move_to = match body.get(i).and_then(|l| l.strip_prefix("*** Move to: ")) {
                Some(to) => {
                    i += 1;
                    Some(to.trim().to_string())
                }
                None => None,
            };
            let mut hunks: Vec<Hunk> = Vec::new();
            while i < body.len() && !is_header(body[i]) {
                let mut hunk = Hunk::default();
                while let Some(anchor) = body.get(i).and_then(|l| l.strip_prefix("@@")) {
                    let anchor = anchor.trim();
                    if !anchor.is_empty() {
                        hunk.anchors.push(anchor.to_string());
                    }
                    i += 1;
                }
                let mut before_lines: Vec<&str> = Vec::new();
                let mut after_lines: Vec<&str> = Vec::new();
                while i < body.len() && !body[i].starts_with("@@") && !is_header(body[i]) {
                    let l = body[i];
                    i += 1;
                    if l.trim_end() == "*** End of File" {
                        hunk.at_eof = true;
                        break;
                    } else if let Some(rest) = l.strip_prefix('+') {
                        after_lines.push(rest);
                    } else if let Some(rest) = l.strip_prefix('-') {
                        before_lines.push(rest);
                    } else if let Some(rest) = l.strip_prefix(' ') {
                        before_lines.push(rest);
                        after_lines.push(rest);
                    } else if l.is_empty() {
                        before_lines.push("");
                        after_lines.push("");
                    } else {
                        anyhow::bail!("unrecognized hunk line: {l:?}");
                    }
                }
                if before_lines.is_empty() && after_lines.is_empty() {
                    anyhow::bail!("empty hunk for {path}");
                }
                hunk.before = before_lines.join("\n");
                hunk.after = after_lines.join("\n");
                hunks.push(hunk);
            }
            ops.push(PatchOp::Update {
                path: path.trim().to_string(),
                move_to,
                hunks,
            });
        } else {
            anyhow::bail!("unexpected line in patch: {line:?}");
        }
    }
    Ok(ops)
}

/// `content` with `hunks` applied in order, each matched exactly once after its
/// anchors. A hunk with no context or removed lines inserts its lines after the
/// last anchor's line, or at the end of the file without one.
fn apply_hunks(path: &str, mut content: String, hunks: &[Hunk]) -> anyhow::Result<String> {
    for (i, hunk) in hunks.iter().enumerate() {
        let n = i + 1;
        let mut from = 0;
        for anchor in &hunk.anchors {
            let Some(at) = content[from..].find(anchor.as_str()) else {
                anyhow::bail!("hunk #{n}: anchor {anchor:?} not found in {path}");
            };
            let line_end = content[from + at..]
                .find('\n')
                .map_or(content.len(), |e| from + at + e + 1);
            from = line_end;
        }

        if hunk.before.is_empty() {
            let at = if hunk.anchors.is_empty() || hunk.at_eof {
                if !content.is_empty() && !content.ends_with('\n') {
                    content.push('\n');
                }
                content.len()
            } else {
                from
            };
            content.insert_str(at, &format!("{}\n", hunk.after));
            continue;
        }

        let mut found = content[from..]
            .match_indices(hunk.before.as_str())
            .map(|(at, _)| from + at);
        let at = if hunk.at_eof {
            let end = content.trim_end_matches('\n').len();
            found
                .filter(|at| at + hunk.before.len() == end)
                .last()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "hunk #{n} not found at the end of {path}; expected:\n{}",
                        hunk.before
                    )
                })?
        } else {
            let Some(at) = found.next() else {
                anyhow::bail!("hunk #{n} not found in {path}; expected:\n{}", hunk.before);
            };
            let rest = found.count();
            if rest > 0 {
                anyhow::bail!(
                    "hunk #{n} matches {} locations in {path}; need more context or an @@ anchor",
                    rest + 1
                );
            }
            at
        };
        content.replace_range(at..at + hunk.before.len(), &hunk.after);
    }
    Ok(content)
}

/// Write `bytes` to `path`, making its directories if they are not there yet.
async fn write_creating_dirs(sandbox: &Sandbox, path: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
    crate::console::write(sandbox, path, bytes).await?;
    Ok(())
}

async fn remove(sandbox: &Sandbox, path: &str) -> anyhow::Result<()> {
    let result = crate::console::exec(sandbox, ["rm", "-f", path], None).await?;
    if result.code != 0 {
        anyhow::bail!("rm {path}: {}", result.stderr.trim());
    }
    Ok(())
}

async fn apply_op(op: &PatchOp, sandbox: &Sandbox) -> anyhow::Result<String> {
    match op {
        // `Add` and a move may name a path whose directories are not there yet.
        PatchOp::Add { path, content } => {
            write_creating_dirs(sandbox, path, content.as_bytes().to_vec()).await?;
            Ok(format!("added {path}"))
        }
        PatchOp::Delete { path } => {
            remove(sandbox, path).await?;
            Ok(format!("deleted {path}"))
        }
        PatchOp::Update {
            path,
            move_to,
            hunks,
        } => {
            let content = String::from_utf8(crate::console::read(sandbox, path).await?)
                .map_err(|_| anyhow::anyhow!("file {path} is not valid UTF-8"))?;
            let content = apply_hunks(path, content, hunks)?;
            match move_to {
                Some(to) => {
                    write_creating_dirs(sandbox, to, content.into_bytes()).await?;
                    remove(sandbox, path).await?;
                    Ok(format!("updated {path}, moved to {to}"))
                }
                None => {
                    crate::console::write(sandbox, path, content).await?;
                    Ok(format!("updated {path}"))
                }
            }
        }
    }
}

/// `apply_patch` after Codex's function (JSON) variant, the one it gives models
/// without freeform tools; the description is Codex's.
pub fn get_apply_patch_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("apply_patch")
        .description(r#"Use the `apply_patch` tool to edit files.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	 def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
"#)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The entire contents of the apply_patch command"
                }
            },
            "required": ["input"],
            "additionalProperties": false
        }))
        .build()
}

pub fn get_apply_patch_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, sandbox: &Sandbox| -> Value {
        let Some(patch_text) = args.pointer("/input").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: input",
                "phase": "validation",
            });
        };

        let ops = match parse_patch(patch_text) {
            Ok(o) => o,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("parse: {e}"),
                    "phase": "parse",
                });
            }
        };

        let mut summary: Vec<Value> = Vec::new();
        for op in &ops {
            match apply_op(op, sandbox).await {
                Ok(msg) => summary.push(Value::from(msg)),
                Err(e) => {
                    return crate::to_value!({
                        "error": format!("{e}"),
                        "applied": Value::Array(summary),
                        "phase": "apply",
                    });
                }
            }
        }
        crate::to_value!({
            "ok": true,
            "applied": Value::Array(summary),
        })
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{console::Sandbox, message::Message, test_sandbox, to_value};

    /// Where a test puts the file a call works on.
    const PATH: &str = "/tmp/subject";

    async fn call(sandbox: &Sandbox, args: Value) -> Message {
        let f = get_apply_patch_tool_func();
        f.call(args, "1", sandbox).next().await.unwrap().message
    }

    #[tokio::test]
    async fn test_apply_patch_add_file() {
        // In directories that are not there yet, which the tool makes.
        let path = "/tmp/new/dir/new.txt";
        let patch = format!("*** Begin Patch\n*** Add File: {path}\n+hello\n+world\n*** End Patch");
        let sandbox = test_sandbox().await;
        let msg = call(&sandbox, to_value!({ "input": patch })).await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "result: {:?}",
            msg.contents[0]
        );
        assert_eq!(
            crate::console::read(&sandbox, path).await.unwrap(),
            b"hello\nworld\n"
        );
    }

    #[tokio::test]
    async fn test_apply_patch_update_file() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "alpha\nbeta\ngamma\n")
            .await
            .unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n alpha\n-beta\n+BETA\n gamma\n*** End Patch",
            PATH
        );
        let msg = call(&sandbox, to_value!({ "input": patch })).await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "result: {:?}",
            msg.contents[0]
        );
        assert_eq!(
            String::from_utf8(crate::console::read(&sandbox, PATH).await.unwrap()).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
    }

    #[tokio::test]
    async fn test_apply_patch_delete_file() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "doomed")
            .await
            .unwrap();
        let patch = format!("*** Begin Patch\n*** Delete File: {PATH}\n*** End Patch");
        let msg = call(&sandbox, to_value!({ "input": patch })).await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        );
        let gone = crate::console::read(&sandbox, PATH).await.unwrap_err();
        assert_eq!(gone.status(), Some(404), "{gone}");
    }

    #[tokio::test]
    async fn test_apply_patch_missing_envelope() {
        let msg = call(
            &test_sandbox().await,
            to_value!({ "input": "*** Add File: foo\n+x" }),
        )
        .await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "parse");
    }

    #[tokio::test]
    async fn test_apply_patch_hunk_not_found() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "alpha\nbeta\n")
            .await
            .unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n nonexistent\n-beta\n+BETA\n*** End Patch",
            PATH
        );
        let msg = call(&sandbox, to_value!({ "input": patch })).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "apply");
        assert_eq!(
            String::from_utf8(crate::console::read(&sandbox, PATH).await.unwrap()).unwrap(),
            "alpha\nbeta\n",
            "file must be unchanged when hunk fails"
        );
    }

    #[tokio::test]
    async fn test_apply_patch_multi_hunk_in_one_file() {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, PATH, "a\nb\nc\nd\ne\n")
            .await
            .unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n-a\n+A\n@@\n-d\n+D\n*** End Patch",
            PATH
        );
        let msg = call(&sandbox, to_value!({ "input": patch })).await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            "result: {:?}",
            msg.contents[0]
        );
        assert_eq!(
            String::from_utf8(crate::console::read(&sandbox, PATH).await.unwrap()).unwrap(),
            "A\nb\nc\nD\ne\n"
        );
    }

    fn patch_file(content: &str, patch: &str) -> anyhow::Result<String> {
        let ops = parse_patch(patch)?;
        let [PatchOp::Update { path, hunks, .. }] = &ops[..] else {
            panic!("expected one Update, got {ops:?}");
        };
        apply_hunks(path, content.to_string(), hunks)
    }

    #[test]
    fn test_apply_patch_anchors_pick_the_occurrence() {
        let content =
            "class A:\n    def f():\n        x = 1\nclass B:\n    def f():\n        x = 1\n";
        let patch = "*** Begin Patch\n*** Update File: m.py\n@@ class B:\n@@     def f():\n-        x = 1\n+        x = 2\n*** End Patch";
        assert_eq!(
            patch_file(content, patch).unwrap(),
            "class A:\n    def f():\n        x = 1\nclass B:\n    def f():\n        x = 2\n"
        );
        let ambiguous = "*** Begin Patch\n*** Update File: m.py\n@@\n-        x = 1\n+        x = 2\n*** End Patch";
        assert!(patch_file(content, ambiguous).is_err());
    }

    #[test]
    fn test_apply_patch_end_of_file_and_insertion() {
        let content = "a\nb\na\n";
        let patch =
            "*** Begin Patch\n*** Update File: f\n@@\n-a\n+A\n*** End of File\n*** End Patch";
        assert_eq!(patch_file(content, patch).unwrap(), "a\nb\nA\n");

        let insert = "*** Begin Patch\n*** Update File: f\n@@ b\n+inserted\n*** End Patch";
        assert_eq!(patch_file(content, insert).unwrap(), "a\nb\ninserted\na\n");

        let append = "*** Begin Patch\n*** Update File: f\n@@\n+tail\n*** End Patch";
        assert_eq!(patch_file("a", append).unwrap(), "a\ntail\n");
    }

    #[test]
    fn test_apply_patch_parses_move_to() {
        let ops = parse_patch(
            "*** Begin Patch\n*** Update File: src/app.py\n*** Move to: src/main.py\n@@ def greet():\n-print(\"Hi\")\n+print(\"Hello, world!\")\n*** Delete File: obsolete.txt\n*** End Patch",
        )
        .unwrap();
        assert!(matches!(
            &ops[..],
            [PatchOp::Update { move_to: Some(to), .. }, PatchOp::Delete { .. }] if to == "src/main.py"
        ));
    }
}
