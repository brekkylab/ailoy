//! The tools a spec brings without naming them: a sandbox's, in the style of the model's org,
//! and a browser's.

/// The org a desc names (`"anthropic"` of `"anthropic/claude-sonnet-5-5"`) and the name
/// after it.
pub(super) fn split_desc(desc: &str) -> (&str, &str) {
    desc.split_once('/').unwrap_or((desc, ""))
}

/// The tools that drive the sandbox's browser.
pub(super) fn browser_tool_names() -> Vec<&'static str> {
    vec!["browser", "browser_screenshot"]
}

/// The shell and file tools for a sandbox, in the style of the model's org.
pub(super) fn sandbox_tool_names(org: &str) -> Vec<&'static str> {
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
