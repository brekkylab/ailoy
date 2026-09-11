//! The system preamble. ailoy sends only what `instruction` says, so this is where the
//! agent learns where it stands and what is mounted there.

use std::path::Path;

use crate::types::{MountInfo, MountKind};

pub struct PromptInput<'a> {
    pub workfs_path: &'a Path,
    pub mounts: &'a [MountInfo],
    pub today: &'a str,
    pub os: &'a str,
    pub extra: Option<&'a str>,
}

pub fn build(input: &PromptInput) -> String {
    let mut s = String::new();
    s.push_str("You are Ailoy, a desktop assistant that works inside the user's workspace: a directory tree the user assembled from local folders and connected services. You read, write and run commands there with your tools, and you explain what you did in plain language.\n\n");
    s.push_str(&format!(
        "Today is {}. The host OS is {}.\n\n",
        input.today, input.os
    ));
    s.push_str(&format!(
        "# Workspace\n\nThe workspace root is `{}`. It is also the shell's working directory, so relative paths resolve inside it. Stay inside the workspace unless the user explicitly asks about another path.\n\n",
        input.workfs_path.display()
    ));
    s.push_str("## Mounts\n\n");
    for m in input.mounts {
        let access = if m.writable {
            "read-write"
        } else {
            "read-only"
        };
        let hint = match m.kind {
            MountKind::Root => "the workspace's own files",
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
        s.push_str(&format!("- `{path}` — {} ({access}): {hint}\n", m.label));
    }
    s.push_str("\nA read-only mount rejects writes; do not retry them — tell the user.\n\n");
    s.push_str("# Tools\n\n`shell` runs `sh -c` in the workspace (output over 30k characters is middle-truncated and flagged `truncated`; a command past its timeout is killed and reported `timed_out`). Prefer `read`, `write`, `edit`, `glob`, `grep` for files, and `shell` for everything else. Run independent tool calls in parallel when it saves time.\n");
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

    #[test]
    fn preamble_names_workfs_mounts_and_readonly() {
        let mounts = vec![
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
        ];
        let s = build(&PromptInput {
            workfs_path: Path::new("/tmp/ws"),
            mounts: &mounts,
            today: "2026-09-11",
            os: "macos",
            extra: Some("Answer in Korean."),
        });
        assert!(s.contains("/tmp/ws"));
        assert!(s.contains("/notion"));
        assert!(s.contains("read-only"));
        assert!(s.contains("page.json"));
        assert!(s.contains("2026-09-11"));
        assert!(s.ends_with("Answer in Korean."));
    }
}
