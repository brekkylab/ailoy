//! The system preamble. ailoy sends only what `instruction` says, so this is where the
//! agent learns where it stands and what is mounted there.

use std::path::Path;

use crate::types::{MountInfo, MountKind, MountStatus};

pub struct PromptInput<'a> {
    pub workfs_path: &'a Path,
    /// Where this run's output belongs, as the agent must spell it.
    ///
    /// The host path, not the one inside the workspace: the workspace is the session's
    /// context and cortex refuses a write anywhere under it. The same files show up in the
    /// workspace at `/artifacts`, which is where the user looks for them.
    pub artifacts_path: &'a Path,
    pub mounts: &'a [MountInfo],
    pub today: &'a str,
    pub os: &'a str,
    /// The session's `provider/model` id. It decides which toolset ailoy attaches — see
    /// `AgentSpec::system_tools` — and so which tools the preamble is allowed to name.
    pub model: &'a str,
    /// The workspace is not mounted in this session: the console stands in the root
    /// directory itself, and nothing attached under a connector path is reachable from it.
    pub degraded: bool,
    pub extra: Option<&'a str>,
}

pub fn build(input: &PromptInput) -> String {
    let mut s = String::new();
    s.push_str("You are Ailoy, a desktop assistant that works inside the user's workspace: a directory tree the user assembled from local folders and connected services. You read it, produce files of your own, run commands, and explain what you did in plain language.\n\n");
    s.push_str(&format!(
        "Today is {}. The host OS is {}.\n\n",
        input.today, input.os
    ));
    s.push_str(&format!(
        concat!(
            "# Workspace\n\n",
            "The workspace root is `{}`. This is the user's tree — what they put there and what their",
            " connectors expose — and it is **read-only to you**. It is managed outside this",
            " conversation: it was there before this run and it stays after. Read it freely; a write",
            " into it is refused.\n\n",
            "# Where your own files go\n\n",
            "Write what you produce under `{}`. That is this run's output, and it is the one tree here",
            " you may write. The user sees the same files inside their workspace at `/{}`, so anything",
            " you leave there is delivered — name it by the path above when you write, and by the",
            " workspace path when you tell the user where it is.\n\n",
            "Your shell starts in neither: it starts in a scratch directory that is thrown away when",
            " this run ends. Use it for intermediates — downloads, unpacked archives, anything you",
            " write only to read back — and put nothing there that the user is meant to keep. Because",
            " that is where you stand, a relative path is always the scratch: name the workspace and",
            " your output by the paths above.\n\n",
        ),
        input.workfs_path.display(),
        input.artifacts_path.display(),
        crate::workspace::ARTIFACTS_PATH,
    ));
    // Every one of these is inside the workspace, so every one of them is read-only to the
    // agent whatever the row says — `writable` is the *user's* access, which is what the file
    // panel shows and what a connector's own store enforces. Telling the agent a folder is
    // read-write here would contradict the section above and cost a refused write to learn.
    s.push_str("## What is in the workspace\n\n");
    for m in input.mounts {
        let hint = match m.kind {
            MountKind::Root => "the user's own files",
            MountKind::Local => "a folder on this computer",
            MountKind::Notion => {
                "a Notion workspace; each page is a directory whose `page.json` holds the page as JSON; databases are directories of pages"
            }
            MountKind::S3 => "an S3 bucket; keys appear as files and directories",
        };
        let path = if m.path.is_empty() {
            "/"
        } else {
            m.path.as_str()
        };
        let label = &m.label;
        // Without the FUSE-T mount the tools see the root directory on disk, not the
        // composed tree: a connector is still configured, still listed — and still not
        // there. Saying so is cheaper than the turns the agent would spend finding out.
        if input.degraded && !matches!(m.kind, MountKind::Root) {
            s.push_str(&format!(
                "- `{path}` — {label} (unavailable: the workspace is not mounted in this session, so this connector is not visible to your tools): {hint}\n"
            ));
            continue;
        }
        match &m.status {
            MountStatus::Ok => s.push_str(&format!("- `{path}` — {label}: {hint}\n")),
            // A mount that failed to come up is listed so the agent knows the path is
            // spoken for, and told why it will not answer, so it does not spend turns
            // finding out.
            MountStatus::Error { message } => s.push_str(&format!(
                "- `{path}` — {label} (unavailable: {message}): {hint}\n"
            )),
        }
    }
    if input.degraded {
        s.push_str(
            "\nOnly the workspace's root directory is reachable in this session. The connector paths above do not exist for your tools; do not try to read or write under them — tell the user the workspace is not mounted.\n\n",
        );
    } else {
        s.push_str("\nA read-only mount rejects writes; do not retry them — tell the user.\n\n");
    }
    s.push_str("# Tools\n\n`shell` runs `sh -c` in the workspace (output over 30k characters is middle-truncated, with the omission marked inline; a command past its timeout is killed and reported `timed_out`, and anything it had written is lost). ");
    // Exactly the tools `AgentSpec::system_tools` attaches for this model family. Naming a
    // tool the model was not given is worse than naming none: it spends a turn calling
    // something that is not in its schema and gets an error back instead of an answer.
    if input.model.starts_with("openai/") {
        s.push_str(
            "Prefer `read` for files, `apply_patch` for edits, and `shell` for everything else. ",
        );
    } else {
        s.push_str(
            "Prefer `read`, `write`, `edit`, `glob`, `grep` for files, and `shell` for everything else. ",
        );
    }
    s.push_str("`web_search` finds pages on the internet and `web_fetch` retrieves one you already have a URL for. Run independent tool calls in parallel when it saves time.\n");
    if let Some(extra) = input.extra.map(str::trim).filter(|e| !e.is_empty()) {
        s.push_str("\n# Additional instructions\n\n");
        s.push_str(extra);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MountInfo, MountKind, MountStatus};

    fn mounts() -> Vec<MountInfo> {
        vec![
            MountInfo {
                id: "r".into(),
                path: "/".into(),
                kind: MountKind::Root,
                label: "Workspace".into(),
                detail: "".into(),
                writable: true,
                status: MountStatus::Ok,
            },
            MountInfo {
                id: "n".into(),
                path: "/notion".into(),
                kind: MountKind::Notion,
                label: "notion".into(),
                detail: "Notion".into(),
                writable: false,
                status: MountStatus::Ok,
            },
            MountInfo {
                id: "s".into(),
                path: "/bucket".into(),
                kind: MountKind::S3,
                label: "bucket".into(),
                detail: "s3://b".into(),
                writable: false,
                status: MountStatus::Error {
                    message: "bucket unreachable".into(),
                },
            },
        ]
    }

    /// A healthy, non-OpenAI session: the defaults every other test varies one field from.
    fn input<'a>(mounts: &'a [MountInfo], model: &'a str, degraded: bool) -> PromptInput<'a> {
        PromptInput {
            workfs_path: Path::new("/tmp/ws"),
            artifacts_path: Path::new("/tmp/out"),
            mounts,
            today: "2026-09-11",
            os: "macos",
            model,
            degraded,
            extra: None,
        }
    }

    #[test]
    fn preamble_names_workfs_mounts_and_readonly() {
        let mounts = mounts();
        let s = build(&PromptInput {
            extra: Some("Answer in Korean."),
            ..input(&mounts, "anthropic/claude-opus-5", false)
        });
        assert!(s.contains("/tmp/ws"));
        assert!(s.contains("/notion"));
        assert!(s.contains("`/bucket` — bucket (unavailable: bucket unreachable)"));
        assert!(s.contains("read-only"));
        assert!(s.contains("page.json"));
        assert!(s.contains("2026-09-11"));
        assert!(s.contains("omission marked inline"));
        assert!(
            !s.contains("flagged `truncated`"),
            "the shell tool sets no such flag for the 30k cut"
        );
        assert!(s.ends_with("Answer in Korean."));
    }

    /// The tool paragraph has to match what ailoy actually attached, and
    /// `AgentSpec::system_tools` gives the two families different toolsets. Naming
    /// `glob`/`grep`/`write`/`edit` to an OpenAI model — or `apply_patch` to any other —
    /// invites a call the model's schema cannot carry.
    #[test]
    fn the_tool_paragraph_names_only_the_tools_this_family_was_given() {
        let mounts = mounts();

        let openai = build(&input(&mounts, "openai/gpt-5", false));
        assert!(openai.contains("`apply_patch`"), "{openai}");
        for absent in ["`glob`", "`grep`", "`write`", "`edit`"] {
            assert!(
                !openai.contains(absent),
                "an openai/* preamble named {absent}, which `system_tools` did not attach"
            );
        }

        let other = build(&input(&mounts, "anthropic/claude-opus-5", false));
        for present in ["`glob`", "`grep`", "`write`", "`edit`", "`read`"] {
            assert!(other.contains(present), "{other}");
        }
        assert!(
            !other.contains("`apply_patch`"),
            "apply_patch is an openai/* tool only"
        );

        // `run.rs` attaches these two for every model, so both renderings say so.
        for s in [&openai, &other] {
            assert!(
                s.contains("`web_search`") && s.contains("`web_fetch`"),
                "{s}"
            );
            assert!(s.contains("`shell`"), "{s}");
        }
    }

    /// Without the mount, the tools stand in the root directory on disk: the connectors are
    /// configured but unreachable, and the preamble says so rather than letting the agent
    /// discover it one failed `ls` at a time.
    #[test]
    fn a_degraded_workspace_marks_every_connector_unreachable() {
        let mounts = mounts();
        let s = build(&input(&mounts, "anthropic/claude-opus-5", true));
        assert!(
            s.contains(
                "- `/notion` — notion (unavailable: the workspace is not mounted in this session, so this connector is not visible to your tools)"
            ),
            "{s}"
        );
        assert!(
            s.contains("- `/bucket` — bucket (unavailable: the workspace is not mounted"),
            "a connector that also failed to build is still reported as unmounted: {s}"
        );
        // The root is the one mount that *is* reachable — it is the tree the console reads.
        assert!(s.contains("- `/` — Workspace: the user's own files"), "{s}");
        assert!(
            s.contains("Only the workspace's root directory is reachable"),
            "{s}"
        );
        assert!(
            !s.contains("A read-only mount rejects writes"),
            "the read-only rule is about mounts that are there: {s}"
        );
    }
}
