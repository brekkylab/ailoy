use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::runenv::{Dirent, FileEntry, RunEnv};

/// Conventional basename of a skill's declaration file.
pub const SKILL_FILE: &str = "SKILL.md";

/// Metadata for one skill (a directory containing `SKILL.md`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

impl SkillMeta {
    pub fn skill_md_path(&self) -> PathBuf {
        self.dir.join(SKILL_FILE)
    }
}

/// Parse `---\nname: …\ndescription: …\n---\n<body>` frontmatter.
pub(crate) fn parse_skill_frontmatter(raw: &str) -> anyhow::Result<(String, String, String)> {
    let after_open = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
        .ok_or_else(|| anyhow::anyhow!("missing leading '---' frontmatter delimiter"))?;
    let (front, rest) = after_open
        .split_once("\n---\n")
        .or_else(|| after_open.split_once("\r\n---\r\n"))
        .or_else(|| after_open.split_once("\n---\r\n"))
        .or_else(|| after_open.split_once("\r\n---\n"))
        .ok_or_else(|| anyhow::anyhow!("missing closing '---' frontmatter delimiter"))?;

    let mut name = None;
    let mut description = None;
    for line in front.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(unquote(v.trim()).to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(unquote(v.trim()).to_string());
        }
    }

    Ok((
        name.ok_or_else(|| anyhow::anyhow!("frontmatter missing `name`"))?,
        description.ok_or_else(|| anyhow::anyhow!("frontmatter missing `description`"))?,
        rest.to_string(),
    ))
}

fn unquote(s: &str) -> &str {
    let trimmed = s.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        if (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
        {
            return &trimmed[1..trimmed.len() - 1];
        }
    }
    trimmed
}

/// Render the "Available Skills" block for an agent's system instruction.
/// Returns `None` when both `skills` is empty and `skill_root` is `None` —
/// in that case the agent has no skill machinery and the block adds noise.
///
/// When `skill_root` is `Some`, the block tells the model that any new
/// skills it creates **must** live under that path so snapshot's auto-
/// discovery picks them up.  When `None`, no creation instructions are
/// rendered — the agent is told what skills exist and nothing else.  Any
/// snapshot-persistence limitations are an internal concern.
pub fn render_skills_table(skills: &[SkillMeta], skill_root: Option<&Path>) -> Option<String> {
    if skills.is_empty() && skill_root.is_none() {
        return None;
    }
    let mut out = String::from(
        "## Available Skills\n\
         Each skill is a directory containing `SKILL.md` (with `name:` and \
         `description:` frontmatter) plus any supporting files (other \
         markdown, scripts, etc.). Read the `SKILL.md` with `cat <path>` \
         before following its steps.\n\n",
    );
    if !skills.is_empty() {
        out.push_str("| Name | Description | SKILL.md |\n|------|-------------|----------|\n");
        for s in skills {
            out.push_str(&format!(
                "| {} | {} | {} |\n",
                s.name,
                s.description,
                s.skill_md_path().display()
            ));
        }
        out.push('\n');
    }
    if let Some(root) = skill_root {
        out.push_str(&format!(
            "**Creating new skills at runtime:** new skills MUST be created \
             under `{}/<skill_name>/`, with a `SKILL.md` carrying `name:` \
             and `description:` frontmatter (plus any supporting files).\n",
            root.display()
        ));
    }
    Some(out)
}

/// Read each declared skill directory's `SKILL.md` from the runenv and
/// return the parsed metadata.  Silently skips entries whose `SKILL.md` is
/// missing or malformed — the skill name (last path segment) is used as-is.
pub async fn discover_skills(
    runenv: &dyn RunEnv,
    skill_dirs: &[PathBuf],
) -> anyhow::Result<Vec<SkillMeta>> {
    let mut out = Vec::new();
    for dir in skill_dirs {
        let Some(meta) = read_skill_meta(runenv, dir).await else {
            continue;
        };
        out.push(meta);
    }
    Ok(out)
}

/// Scan a single fixed directory for `<child>/SKILL.md` entries that are
/// not already declared.  Used by [`Agent::snapshot`](crate::agent::Agent::snapshot)
/// to round-trip skills the agent created at runtime under the spec's
/// `skill_root`.  Single `ls` call — O(children of `skill_root`).
pub async fn discover_new_skills(
    runenv: &dyn RunEnv,
    skill_root: &Path,
    declared: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    let entries = match runenv.ls(skill_root).await {
        Ok(es) => es,
        Err(_) => return Ok(Vec::new()),
    };
    let declared_set: std::collections::HashSet<&PathBuf> = declared.iter().collect();
    let mut out = Vec::new();
    for entry in entries {
        let Dirent::Dir { name, .. } = entry else {
            continue;
        };
        let candidate = skill_root.join(&name);
        if declared_set.contains(&candidate) {
            continue;
        }
        if runenv.read(&candidate.join(SKILL_FILE)).await.is_ok() {
            out.push(candidate);
        }
    }
    Ok(out)
}

async fn read_skill_meta(runenv: &dyn RunEnv, dir: &Path) -> Option<SkillMeta> {
    let bytes = runenv.read(&dir.join(SKILL_FILE)).await.ok()?;
    let raw = String::from_utf8(bytes).ok()?;
    let (_n, description, _body) = parse_skill_frontmatter(&raw).ok()?;
    let name = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(SkillMeta {
        name,
        description,
        dir: dir.to_path_buf(),
    })
}

/// Parse declared skills' `SKILL.md` from the in-memory file list.
/// Missing files are skipped; malformed frontmatter is an error.
pub fn scan_declared_skills(
    files: &[FileEntry],
    skill_dirs: &[PathBuf],
) -> anyhow::Result<Vec<SkillMeta>> {
    let mut out = Vec::new();
    for dir in skill_dirs {
        let skill_md = dir.join(SKILL_FILE);
        let Some(file) = files.iter().find(|f| f.path == skill_md) else {
            continue;
        };
        let raw = std::str::from_utf8(file.content.as_ref()).map_err(|e| {
            anyhow::anyhow!("SKILL.md at {} is not valid UTF-8: {e}", skill_md.display())
        })?;
        let (name, description, _body) = parse_skill_frontmatter(raw)
            .map_err(|e| anyhow::anyhow!("SKILL.md at {}: {e}", skill_md.display()))?;
        out.push(SkillMeta {
            name,
            description,
            dir: dir.clone(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::runenv::Local;

    /// Build a `dyn RunEnv` backed by a real [`Local`].  Tests pass absolute
    /// paths *under the supplied tempdir* directly to `materialise_skills` /
    /// `discover_skills` etc., so no path translation is needed.
    fn local_runenv() -> Arc<dyn RunEnv> {
        Arc::new(Local {})
    }

    #[test]
    fn test_parse_frontmatter_basic() {
        let raw = "---\nname: foo\ndescription: do the foo thing\n---\nbody here\n";
        let (name, desc, body) = parse_skill_frontmatter(raw).unwrap();
        assert_eq!(name, "foo");
        assert_eq!(desc, "do the foo thing");
        assert_eq!(body, "body here\n");
    }

    #[test]
    fn test_parse_frontmatter_quoted() {
        let raw = "---\nname: \"foo bar\"\ndescription: 'quoted desc'\n---\nx\n";
        let (name, desc, _) = parse_skill_frontmatter(raw).unwrap();
        assert_eq!(name, "foo bar");
        assert_eq!(desc, "quoted desc");
    }

    #[test]
    fn test_parse_frontmatter_missing_keys() {
        assert!(parse_skill_frontmatter("---\nname: foo\n---\nbody").is_err());
    }

    #[test]
    fn test_parse_frontmatter_no_delimiters() {
        assert!(parse_skill_frontmatter("no frontmatter").is_err());
    }

    #[test]
    fn test_render_skills_table_empty_and_no_root_returns_none() {
        // No skills and no skill_root → block adds no value, omit it.
        assert!(render_skills_table(&[], None).is_none());
    }

    #[test]
    fn test_render_skills_table_empty_skills_but_root_set() {
        // Empty list but skill_root set → still render so the model knows
        // it MAY create skills (and where).
        let out = render_skills_table(&[], Some(Path::new("/workspace/skills"))).unwrap();
        assert!(out.contains("Available Skills"));
        assert!(out.contains("Creating new skills at runtime"));
        assert!(out.contains("/workspace/skills"));
        // No table rows since skills is empty.
        assert!(!out.contains("| Name |"));
    }

    #[test]
    fn test_render_skills_table_basic_with_root() {
        let metas = vec![SkillMeta {
            name: "foo".into(),
            description: "do foo".into(),
            dir: PathBuf::from("/workspace/skills/foo"),
        }];
        let out = render_skills_table(&metas, Some(Path::new("/workspace/skills"))).unwrap();
        assert!(out.contains("Available Skills"));
        assert!(out.contains("| foo | do foo | /workspace/skills/foo/SKILL.md |"));
        // Crucially: the block points at skill_root for new skill creation.
        assert!(
            out.contains("/workspace/skills/<skill_name>/"),
            "block must direct the model to skill_root for new skills: {out}"
        );
        assert!(out.contains("MUST"));
    }

    #[test]
    fn test_render_skills_table_without_root_omits_creation_block() {
        let metas = vec![SkillMeta {
            name: "foo".into(),
            description: "do foo".into(),
            dir: PathBuf::from("/workspace/skills/foo"),
        }];
        let out = render_skills_table(&metas, None).unwrap();
        assert!(out.contains("Available Skills"));
        assert!(out.contains("| foo | do foo |"));
        // No creation instructions and no internal warnings — the agent
        // only learns what skills exist.
        assert!(!out.contains("Creating new skills at runtime"));
        assert!(!out.contains("skill_root"));
    }

    #[test]
    fn test_scan_declared_skills_uses_frontmatter_name() {
        // Crucially: skill `name` comes from frontmatter, NOT the dir's
        // last segment.  Here dir says "greet_dir_name" but frontmatter
        // says "greet" — frontmatter wins.
        let dir = PathBuf::from("/workspace/skills/greet_dir_name");
        let files = vec![FileEntry::new(
            "/workspace/skills/greet_dir_name/SKILL.md",
            b"---\nname: greet\ndescription: say hello\n---\nbody\n".to_vec(),
        )];
        let metas = scan_declared_skills(&files, &[dir.clone()]).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].name, "greet");
        assert_eq!(metas[0].description, "say hello");
        assert_eq!(metas[0].dir, dir);
    }

    #[test]
    fn test_scan_declared_skills_skips_missing_skill_md() {
        // Declared skill but no FileEntry providing the SKILL.md → silently
        // skipped from the rendered table.
        let dir = PathBuf::from("/workspace/skills/orphan");
        assert!(scan_declared_skills(&[], &[dir]).unwrap().is_empty());
    }

    #[test]
    fn test_scan_declared_skills_errors_on_malformed_frontmatter() {
        let bad_dir = PathBuf::from("/workspace/skills/bad");
        let files = vec![FileEntry::new(
            "/workspace/skills/bad/SKILL.md",
            b"no frontmatter at all\n".to_vec(),
        )];
        assert!(scan_declared_skills(&files, &[bad_dir]).is_err());
    }

    #[test]
    fn test_scan_declared_skills_errors_on_missing_name() {
        let dir = PathBuf::from("/workspace/skills/anon");
        let files = vec![FileEntry::new(
            "/workspace/skills/anon/SKILL.md",
            b"---\ndescription: no name field\n---\nbody\n".to_vec(),
        )];
        assert!(scan_declared_skills(&files, &[dir]).is_err());
    }

    #[tokio::test]
    async fn test_discover_skills_reads_each_declared_dir() {
        let dir = tempfile::tempdir().unwrap();
        let runenv = local_runenv();

        let greet_dir = dir.path().join("skills/greet");
        runenv.mkdir(&greet_dir).await.unwrap();
        runenv
            .write(
                &greet_dir.join("SKILL.md"),
                b"---\nname: greet\ndescription: say hello\n---\nbody\n",
            )
            .await
            .unwrap();
        runenv
            .write(&greet_dir.join("helper.py"), b"# supporting\n")
            .await
            .unwrap();

        let metas = discover_skills(&*runenv, &[greet_dir.clone()])
            .await
            .unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].name, "greet");
        assert_eq!(metas[0].description, "say hello");
        assert_eq!(metas[0].dir, greet_dir);
    }

    #[tokio::test]
    async fn test_discover_skills_missing_dir_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let runenv = local_runenv();
        let metas = discover_skills(&*runenv, &[dir.path().join("does_not_exist")])
            .await
            .unwrap();
        assert!(metas.is_empty());
    }

    #[tokio::test]
    async fn test_discover_new_skills_finds_sibling_with_skill_md() {
        let dir = tempfile::tempdir().unwrap();
        let runenv = local_runenv();
        let root = dir.path().join("skills");
        let declared = root.join("declared");
        let runtime = root.join("runtime");

        runenv.mkdir(&declared).await.unwrap();
        runenv
            .write(
                &declared.join("SKILL.md"),
                b"---\nname: declared\ndescription: d\n---\nb\n",
            )
            .await
            .unwrap();
        runenv.mkdir(&runtime).await.unwrap();
        runenv
            .write(
                &runtime.join("SKILL.md"),
                b"---\nname: runtime\ndescription: r\n---\nb\n",
            )
            .await
            .unwrap();

        let news = discover_new_skills(&*runenv, &root, &[declared.clone()])
            .await
            .unwrap();
        assert_eq!(news, vec![runtime]);
    }

    #[tokio::test]
    async fn test_discover_new_skills_skips_dirs_without_skill_md() {
        let dir = tempfile::tempdir().unwrap();
        let runenv = local_runenv();
        let root = dir.path().join("skills");

        let noskill = root.join("noskill");
        runenv.mkdir(&noskill).await.unwrap();
        runenv
            .write(&noskill.join("notes.md"), b"# nothing special\n")
            .await
            .unwrap();

        let news = discover_new_skills(&*runenv, &root, &[]).await.unwrap();
        assert!(news.is_empty());
    }

    /// End-to-end verification with a small, fast-running skill: a Python
    /// micro-benchmark helper.  Only requires `python3` inside the sandbox.
    ///
    /// Flow:
    /// 1. `AgentBuilder` declares `benchmark_python_snippet` at
    ///    `/workspace/skills/benchmark_python_snippet/` and seeds its
    ///    SKILL.md (with a YAML frontmatter naming the skill).
    /// 2. The auto-rendered "Available Skills" table in the system
    ///    instruction surfaces the skill to the model.
    /// 3. The user asks to compare two Python expressions — this query
    ///    matches the skill's `description` so the model should `cat` the
    ///    SKILL.md, follow its `python -m timeit -v` protocol once per
    ///    expression, and emit a comparison table in the documented
    ///    format.
    ///
    /// Requires `ANTHROPIC_API_KEY` and the `sandbox` feature.  Marked
    /// `#[ignore]` because it still spawns a real sandbox and calls the
    /// Anthropic API.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    #[ignore = "slow: spawns a real sandbox and calls the Anthropic API"]
    async fn test_benchmark_python_snippet_skill() {
        use futures::StreamExt as _;

        use crate::{
            agent::{AgentBuilder, AgentProvider},
            lang_model::LangModelProvider,
            message::{FinishReason, Message, Part, Role},
            runenv::{Sandbox, SandboxConfig},
            tool::ToolProvider,
        };

        let skill_md = r#"---
name: benchmark_python_snippet
description: Benchmark a small Python expression or statement using `python -m timeit` with a fixed `-n` / `-r` protocol, then summarise the run as a markdown table. Use when the user asks "how fast is …", "benchmark this Python snippet", or compares two Python expressions.
---
# Skill: Benchmark Python Snippet

A short, **reproducible** playbook for timing a small piece of Python
inside the sandbox.  Use only `python -m timeit` with the exact flags
listed below — do *not* roll your own `time.time()` loop, and do *not*
fall back to `timeit.timeit()` in a REPL (those drop the per-loop
calibration and produce noisier numbers).

## When to use

- The user names a Python expression or `stmt` they want timed.
- The user asks to compare two expressions head-to-head — run this
  skill once per expression and emit a single combined table.

## Inputs

| Name      | Required | Notes                                                                 |
|-----------|----------|-----------------------------------------------------------------------|
| `stmt`    | yes      | The statement to time — quoted exactly as the user wrote it.          |
| `setup`   | no       | Setup code that runs once per *run* (not per call). Default: `pass`.  |
| `number`  | no       | Per-run loop count `-n N`. Default: let `timeit` autocalibrate (omit `-n`). |
| `repeat`  | no       | Number of runs `-r R`. Default: `5`.                                  |

If the user did not specify `number`/`repeat`, take the defaults and
say so in the output (`**Per-run loops:** auto (default)`,
`**Runs:** 5 (default)`).

## Steps

### 1. Make sure Python is available

```bash
python3 --version >/dev/null 2>&1 \
    || { echo "no python3 in sandbox" 1>&2; exit 2; }
```

If this fails, **stop** and emit only the `## Errors` section of the
template (see "Rules").

### 2. Run the benchmark

Build the command from the inputs.  `python -m timeit` already runs
`R` independent timings of `N` iterations each and prints one line
per run when `-v` is set.

```bash
# Substitute SETUP / STMT / FLAGS.  Quote with single quotes — never
# interpolate the user's text directly into a double-quoted string.
python3 -m timeit -v -r "$REPEAT" -s "$SETUP" $NUMBER_FLAG -- "$STMT"
```

- `NUMBER_FLAG` is `-n $NUMBER` when the user gave a `number`, else
  the empty string (so `timeit` auto-picks).
- `--` separates flags from the statement so a leading `-` in `STMT`
  is not mistaken for a flag.

### 3. Parse the output

`timeit -v` emits one calibration line, several "raw times" lines,
then a final "best of R" line.  Extract three things:

| Field        | grep pattern                                          |
|--------------|-------------------------------------------------------|
| `loops`      | `^([0-9]+) loops, best of`                            |
| `best_str`   | last `best of [0-9]+: (.*) per loop$`                 |
| `raw_times`  | line after `raw times:` — space-separated quantities. |

Normalise every quantity (`raw_times` *and* `best_str`) to **microseconds
per loop**:

| Suffix in timeit output | Multiplier to µs |
|-------------------------|------------------|
| `nsec`                  | `× 0.001`        |
| `usec`                  | `× 1`            |
| `msec`                  | `× 1000`         |
| `sec`                   | `× 1_000_000`    |

Compute `min`, `max`, and `mean` from the normalised `raw_times`.

### 4. Output template

Reply with exactly this markdown.  Do **not** add introductory or
closing prose.

```
# Benchmark — `<one-line of STMT>`

**Setup:** `<SETUP>`{ " (default)" if user did not override }
**Per-run loops:** <loops>{ " (default)" if user did not override `number` }
**Runs:** <repeat>{ " (default)" if user did not override `repeat` }

| Metric | Value (µs / loop) |
|--------|--------------------|
| min    | <min>              |
| mean   | <mean>             |
| max    | <max>              |
| best   | <best>             |

Raw runs (µs / loop): <r1>, <r2>, …, <rR>
```

Render numbers to **3 significant figures** (e.g. `0.142`, `12.3`,
`1230`).  Use the same precision in every cell — no `1e-7` exponents.

## Rules

- Never use `time.time()`, `time.perf_counter()`, or
  `timeit.timeit(...)` in a REPL — they bypass timeit's per-loop
  calibration and we treat their numbers as invalid.
- Never report numbers in mixed units (e.g. "best 12 µs, worst 4 ms").
  Always convert to a single unit (µs / loop) before formatting.
- Never drop the `--` separator — a snippet that starts with `-`
  would otherwise be parsed as a flag.
- Never re-run with different `--repeat` to "smooth out" the result;
  report the actual numbers from one run of the protocol.
- On any error from step 1 or 2, emit only this block and stop:

  ```
  # Benchmark — <one-line of STMT>

  ## Errors

  ```
  <stderr verbatim>
  ```
  ```
"#;

        let instruction = "You are a helpful assistant with access to skills. \
             To activate a skill, read its SKILL.md using the bash tool \
             (`cat <path>`), then follow the instructions inside.";

        let mut provider = AgentProvider::new();
        provider.models.insert(
            "anthropic/*".into(),
            LangModelProvider::anthropic(std::env::var("ANTHROPIC_API_KEY").unwrap()),
        );
        provider.tools = ToolProvider::new();

        let sandbox = Sandbox::new(SandboxConfig {
            image: "python:3.12-slim".into(),
            ..SandboxConfig::default()
        })
        .await
        .expect("sandbox creation failed");

        let skill_dir = PathBuf::from("/workspace/skills/benchmark_python_snippet");
        let mut agent = AgentBuilder::new("anthropic/claude-sonnet-4-6")
            .provider(provider)
            .runenv(sandbox)
            .tools([
                crate::tool::r#impl::get_bash_tool_desc(),
                crate::tool::r#impl::get_python_repl_tool_desc(),
            ])
            .instruction(instruction)
            .skill_root("/workspace/skills")
            .skill(
                &skill_dir,
                [FileEntry::new(
                    skill_dir.join("SKILL.md"),
                    skill_md.as_bytes().to_vec(),
                )],
            )
            .build()
            .unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Compare sum(range(10000)) vs sum(i for i in range(10000)).",
        )]);

        // Capture the agent's final assistant turn.
        let mut final_text = String::new();
        {
            let mut stream = agent.run(query);
            while let Some(event) = stream.next().await {
                let output = event.expect("agent stream error");
                if output.message.role == Role::Assistant
                    && matches!(output.finish_reason, FinishReason::Stop {})
                {
                    final_text = output
                        .message
                        .contents
                        .iter()
                        .filter_map(|p| p.as_text())
                        .collect::<Vec<_>>()
                        .join("");
                }
            }
        }

        // Sanity: the SKILL.md was materialised into the sandbox.
        let skill_md_on_disk = agent
            .state
            .runenv
            .read(&skill_dir.join("SKILL.md"))
            .await
            .expect("SKILL.md should have been materialised");
        let head = std::str::from_utf8(&skill_md_on_disk[..50]).unwrap_or("");
        assert!(
            head.starts_with("---\nname: benchmark_python_snippet"),
            "SKILL.md frontmatter should be preserved, got prefix: {head:?}"
        );

        // The final assistant text should contain the documented benchmark
        // template — at minimum the per-loop unit column and both
        // expressions naming.  This is a soft check: it allows the model
        // some leeway in formatting but enforces that it (a) actually
        // followed the skill's output template and (b) benchmarked both
        // requested expressions.
        assert!(
            final_text.contains("µs / loop") || final_text.contains("us / loop"),
            "expected the benchmark template's per-loop unit column in the \
             final answer:\n{final_text}"
        );
        assert!(
            final_text.contains("sum(range(10000))"),
            "expected first expression to appear in result:\n{final_text}"
        );
        assert!(
            final_text.contains("sum(i for i in range(10000))"),
            "expected second expression to appear in result:\n{final_text}"
        );
    }
}
