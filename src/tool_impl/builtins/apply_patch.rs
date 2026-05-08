use std::path::Path;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    runenv::RunEnv,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

const DESCRIPTION: &str = "Apply a patch to the filesystem.

The patch envelope:

    *** Begin Patch
    <one or more file ops>
    *** End Patch

File operations:

    *** Add File: <path>
    +line 1
    +line 2

    *** Update File: <path>
    @@ <optional anchor — ignored, used as a hint only>
     context line (leading single space)
    -line to remove
    +line to add
     context line

    *** Delete File: <path>

Multiple hunks for one file are separated by additional `@@` lines. \
For Update, the `before` block (context + removed lines, with prefixes \
stripped) must match exactly once in the target file; otherwise the \
patch is rejected with no changes applied.";

#[derive(Debug)]
enum PatchOp {
    Add { path: String, content: String },
    Update { path: String, hunks: Vec<Hunk> },
    Delete { path: String },
}

#[derive(Debug)]
struct Hunk {
    before: String,
    after: String,
}

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

    let mut ops = Vec::new();
    let mut i = 0;
    while i < body.len() {
        let line = body[i];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            i += 1;
            let mut content_lines: Vec<&str> = Vec::new();
            while i < body.len() && !body[i].starts_with("*** ") {
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
            let mut hunks: Vec<Hunk> = Vec::new();
            while i < body.len() && !body[i].starts_with("*** ") {
                if !body[i].starts_with("@@") {
                    anyhow::bail!(
                        "expected '@@' inside Update File hunk, got: {:?}",
                        body[i]
                    );
                }
                i += 1; // skip @@ anchor line
                let mut before_lines: Vec<&str> = Vec::new();
                let mut after_lines: Vec<&str> = Vec::new();
                while i < body.len()
                    && !body[i].starts_with("@@")
                    && !body[i].starts_with("*** ")
                {
                    let l = body[i];
                    if let Some(rest) = l.strip_prefix('+') {
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
                    i += 1;
                }
                if before_lines.is_empty() {
                    anyhow::bail!(
                        "hunk for {path} has no context or removal lines — \
                         pure-insertion hunks are not supported"
                    );
                }
                hunks.push(Hunk {
                    before: before_lines.join("\n"),
                    after: after_lines.join("\n"),
                });
            }
            ops.push(PatchOp::Update {
                path: path.trim().to_string(),
                hunks,
            });
        } else {
            anyhow::bail!("unexpected line in patch: {line:?}");
        }
    }
    Ok(ops)
}

async fn apply_op(op: &PatchOp, runenv: &dyn RunEnv) -> anyhow::Result<String> {
    match op {
        PatchOp::Add { path, content } => {
            runenv.write(Path::new(path), content.as_bytes()).await?;
            Ok(format!("added {path}"))
        }
        PatchOp::Delete { path } => {
            let result = runenv
                .exec("rm".into(), vec!["-f".into(), path.clone()], None)
                .await?;
            if result.exit_code != 0 {
                anyhow::bail!("rm {path}: {}", result.stderr.trim());
            }
            Ok(format!("deleted {path}"))
        }
        PatchOp::Update { path, hunks } => {
            let bytes = runenv.read(Path::new(path)).await?;
            let mut content = String::from_utf8(bytes)
                .map_err(|_| anyhow::anyhow!("file {path} is not valid UTF-8"))?;
            for (i, hunk) in hunks.iter().enumerate() {
                let n = content.matches(&hunk.before).count();
                if n == 0 {
                    anyhow::bail!(
                        "hunk #{} not found in {path}; expected:\n{}",
                        i + 1,
                        hunk.before
                    );
                }
                if n > 1 {
                    anyhow::bail!(
                        "hunk #{} matches {n} locations in {path}; need more context",
                        i + 1
                    );
                }
                content = content.replacen(&hunk.before, &hunk.after, 1);
            }
            runenv.write(Path::new(path), content.as_bytes()).await?;
            Ok(format!("updated {path}"))
        }
    }
}

pub async fn build_apply_patch_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("apply_patch")
        .description(DESCRIPTION)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "The full patch text, including the *** Begin Patch / *** End Patch envelope."
                }
            },
            "required": ["patch"]
        }))
        .build();

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let Some(patch_text) = args.pointer("/patch").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: patch",
                "phase": "validation"
            });
        };

        let ops = match parse_patch(patch_text) {
            Ok(o) => o,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("parse: {e}"),
                    "phase": "parse"
                });
            }
        };

        let mut summary: Vec<Value> = Vec::new();
        for op in &ops {
            match apply_op(op, ctx.runenv.as_ref()).await {
                Ok(msg) => summary.push(Value::from(msg)),
                Err(e) => {
                    return crate::to_value!({
                        "error": format!("{e}"),
                        "applied": Value::Array(summary),
                        "phase": "apply"
                    });
                }
            }
        }
        crate::to_value!({
            "ok": true,
            "applied": Value::Array(summary)
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
    async fn test_apply_patch_add_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        let patch = format!(
            "*** Begin Patch\n*** Add File: {}\n+hello\n+world\n*** End Patch",
            path.display()
        );
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": patch }), local_ctx())
            .await;
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
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello\nworld\n");
    }

    #[tokio::test]
    async fn test_apply_patch_update_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "alpha\nbeta\ngamma\n").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n alpha\n-beta\n+BETA\n gamma\n*** End Patch",
            tmp.path().display()
        );
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": patch }), local_ctx())
            .await;
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
            std::fs::read_to_string(tmp.path()).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
    }

    #[tokio::test]
    async fn test_apply_patch_delete_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "doomed").unwrap();
        let path_str = tmp.path().to_string_lossy().to_string();
        let patch = format!("*** Begin Patch\n*** Delete File: {path_str}\n*** End Patch");
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": patch }), local_ctx())
            .await;
        assert!(
            msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        );
        assert!(!std::path::Path::new(&path_str).exists());
    }

    #[tokio::test]
    async fn test_apply_patch_missing_envelope() {
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": "*** Add File: foo\n+x" }), local_ctx())
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
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "alpha\nbeta\n").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n nonexistent\n-beta\n+BETA\n*** End Patch",
            tmp.path().display()
        );
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": patch }), local_ctx())
            .await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "apply");
        assert_eq!(
            std::fs::read_to_string(tmp.path()).unwrap(),
            "alpha\nbeta\n",
            "file must be unchanged when hunk fails"
        );
    }

    #[tokio::test]
    async fn test_apply_patch_multi_hunk_in_one_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\nb\nc\nd\ne\n").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n-a\n+A\n@@\n-d\n+D\n*** End Patch",
            tmp.path().display()
        );
        let tool = build_apply_patch_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "patch": patch }), local_ctx())
            .await;
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
            std::fs::read_to_string(tmp.path()).unwrap(),
            "A\nb\nc\nD\ne\n"
        );
    }
}
