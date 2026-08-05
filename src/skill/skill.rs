use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::runenv::FileEntry;

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
/// Returns `None` when `skills` is empty.
pub fn render_skills_table(skills: &[SkillMeta]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from(
        "## Available Skills\n\
         Each skill is a directory containing `SKILL.md` (with `name:` and \
         `description:` frontmatter) plus any supporting files (other \
         markdown, scripts, etc.). Read the `SKILL.md` with `cat <path>` \
         before following its steps.\n\n\
         | Name | Description | SKILL.md |\n|------|-------------|----------|\n",
    );
    for s in skills {
        out.push_str(&format!(
            "| {} | {} | {} |\n",
            s.name,
            s.description,
            s.skill_md_path().display()
        ));
    }
    Some(out)
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
    use super::*;

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
    fn test_render_skills_table_empty_returns_none() {
        assert!(render_skills_table(&[]).is_none());
    }

    #[test]
    fn test_render_skills_table_basic() {
        let metas = vec![SkillMeta {
            name: "foo".into(),
            description: "do foo".into(),
            dir: PathBuf::from("/workspace/skills/foo"),
        }];
        let out = render_skills_table(&metas).unwrap();
        assert!(out.contains("Available Skills"));
        assert!(out.contains("| foo | do foo | /workspace/skills/foo/SKILL.md |"));
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
        let metas = scan_declared_skills(&files, std::slice::from_ref(&dir)).unwrap();
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

    /// End-to-end verification with a small, fast-running skill: a Python
    /// micro-benchmark helper. An agent activates `benchmark_python_snippet`
    /// inside a `python:3.12-slim` sandbox, materialises its SKILL.md into the
    /// sandbox, then follows the skill's `python -m timeit` protocol and emits
    /// the documented comparison table.
    ///
    /// `#[ignore]` — boots a real microVM and calls the Anthropic API. Run with:
    /// `ANTHROPIC_API_KEY=… cargo test --features sandbox -- --ignored`. The
    /// kernel and rootfs are resolved by `Sandbox` itself; no manual codesign
    /// (`Sandbox::exec` boots an ad-hoc-signed copy on macOS), and the VM boots
    /// from ailoy's link-time constructor — no `boot_if_requested`.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[ignore = "boots a real microVM and calls the Anthropic API"]
    async fn test_benchmark_python_snippet_skill() {
        use std::sync::Arc;

        use futures::StreamExt as _;

        use crate::{
            agent::{AgentBuilder, AgentProvider, get_agent_providers_mut},
            lang_model::{LangModelProvider, get_lm_providers_mut},
            message::{FinishReason, Message, Part, Role},
            runenv::{Console, Sandbox},
            tool::{
                ToolProvider, get_tool_providers_mut,
                r#impl::{get_python_repl_tool_desc, get_shell_tool_desc},
            },
        };

        let api_key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(k) => k,
            Err(_) => {
                eprintln!("skipped: ANTHROPIC_API_KEY unset");
                return;
            }
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

        let model = std::env::var("SKILL_TEST_MODEL")
            .unwrap_or_else(|_| "anthropic/claude-sonnet-4-6".into());

        let upper = std::env::temp_dir().join("ailoy_skill_bench.img");
        let _ = std::fs::remove_file(&upper);

        const PROVIDER: &str = "skill_benchmark_python_snippet";
        {
            let mut lmp = LangModelProvider::new();
            lmp.insert("anthropic/*".into(), LangModelProvider::anthropic(api_key));
            get_lm_providers_mut().insert(PROVIDER.into(), lmp);
            get_tool_providers_mut().insert(PROVIDER.into(), ToolProvider::new());
            get_agent_providers_mut()
                .insert(PROVIDER.into(), AgentProvider::new(PROVIDER, PROVIDER));
        }

        // No workspace: the kernel and rootfs are resolved by `Sandbox`, and the
        // agent materialises SKILL.md straight into the sandbox filesystem, which
        // persists across the ephemeral per-exec VMs.
        let sandbox = Sandbox::new(&upper)
            .expect("build sandbox")
            .with_image("python:3.12-slim")
            .await
            .expect("pull image rootfs");
        let console: Arc<dyn Console> = Arc::new(sandbox);

        let instruction = "You are a helpful assistant with access to skills. \
             To activate a skill, read its SKILL.md using the shell tool \
             (`cat <path>`), then follow the instructions inside.";
        let skill_dir = PathBuf::from("/root/skills/benchmark_python_snippet");

        let mut agent = AgentBuilder::new(&model)
            .agent_provider(PROVIDER)
            .console(console.clone())
            .tools([get_shell_tool_desc(), get_python_repl_tool_desc()])
            .instruction(instruction)
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

        // The skill was materialised into the sandbox workspace.
        let on_disk = console
            .read(&skill_dir.join("SKILL.md"))
            .await
            .expect("SKILL.md should have been materialised");
        let head = std::str::from_utf8(&on_disk[..on_disk.len().min(50)]).unwrap_or("");
        assert!(
            head.starts_with("---\nname: benchmark_python_snippet"),
            "SKILL.md frontmatter should be preserved, got: {head:?}"
        );

        // The final answer followed the skill's template (per-loop microsecond
        // unit) and named both requested expressions. Ignore whitespace so both
        // "µs / loop" and "µs/loop" pass — the model varies the spacing.
        let compact: String = final_text.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            compact.contains("µs/loop") || compact.contains("us/loop"),
            "expected the benchmark template's per-loop unit column:\n{final_text}"
        );
        assert!(
            final_text.contains("sum(range(10000))"),
            "expected first expression in result:\n{final_text}"
        );
        assert!(
            final_text.contains("sum(i for i in range(10000))"),
            "expected second expression in result:\n{final_text}"
        );
    }
}
