//! The shell and file tools, which run in the agent's sandbox.

mod apply_patch;
mod edit;
mod imgread;
mod read;
mod read_file;
mod shell;
mod write;

pub use apply_patch::*;
pub use edit::*;
pub use imgread::*;
pub use read::*;
pub use read_file::*;
pub use shell::*;
pub use write::*;

use crate::tool::{ToolDesc, ToolFunc, ToolSet};

/// `shell`, `read`, `read_file`, `imgread`, `write`, `edit` and `apply_patch`, in the agent's
/// sandbox.
pub struct Console;

impl ToolSet for Console {
    fn descs(&self) -> Vec<ToolDesc> {
        vec![
            get_shell_tool_desc(),
            get_read_tool_desc(),
            get_read_file_tool_desc(),
            get_imgread_tool_desc(),
            get_write_tool_desc(),
            get_edit_tool_desc(),
            get_apply_patch_tool_desc(),
        ]
    }

    fn make(&self, name: &str) -> anyhow::Result<ToolFunc> {
        Ok(match name {
            "shell" => get_shell_tool_func(),
            "read" => get_read_tool_func(),
            "read_file" => get_read_file_tool_func(),
            "imgread" => get_imgread_tool_func(),
            "write" => get_write_tool_func(),
            "edit" => get_edit_tool_func(),
            "apply_patch" => get_apply_patch_tool_func(),
            _ => anyhow::bail!("'{name}' is no console tool"),
        })
    }
}

/// The console tools a sandbox brings, in the style of the model's org.
pub(crate) fn sandbox_tool_names(org: &str) -> Vec<&'static str> {
    let mut names = vec!["shell"];
    // GPT reads files through `shell`, as in Codex.
    if org == "openai" {
        names.push("apply_patch");
    } else {
        // DeepSeek, Kimi and GLM serve Anthropic-compatible APIs for Claude Code, so
        // likely follow the Claude style.
        names.push(match org {
            "google" => "read_file",
            _ => "read",
        });
        names.push("write");
        names.push("edit");
    }

    // `imgread` only where the image reaches the model: the backends of these take tool
    // results as text alone (an image becomes a label), even for vision models.
    if !matches!(
        org,
        "deepseek-ai" | "moonshotai" | "zai-org" | "Qwen" | "LGAI-EXAONE"
    ) {
        names.push("imgread");
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_tools_follow_the_model() {
        assert_eq!(
            sandbox_tool_names("openai"),
            ["shell", "apply_patch", "imgread"]
        );
        assert!(sandbox_tool_names("google").contains(&"read_file"));

        let claude = sandbox_tool_names("anthropic");
        assert!(claude.contains(&"read"), "{claude:?}");
        assert!(claude.contains(&"imgread"), "{claude:?}");

        assert!(!sandbox_tool_names("deepseek-ai").contains(&"imgread"));
        assert!(!sandbox_tool_names("moonshotai").contains(&"imgread"));
        assert!(!sandbox_tool_names("LGAI-EXAONE").contains(&"imgread"));
    }
}
