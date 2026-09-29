use super::read::load_text;
use crate::{
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_MAX_LINES: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;

fn error_message(id: String, msg: impl Into<String>) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::value(crate::to_value!({ "error": msg.into() }))])
        .with_id(id)
}

/// Lines `start_line..=end_line` (1-based) of `text` as Gemini CLI returns them:
/// the raw text, led by a note on how to read the rest when anything was cut.
fn format_text(text: &str, start_line: Option<usize>, end_line: Option<usize>) -> String {
    let lines: Vec<&str> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    let total = lines.len();

    let slice_start = start_line.map_or(0, |n| n.saturating_sub(1));
    let slice_end = match end_line {
        Some(end) => end.min(total),
        None => (slice_start + DEFAULT_MAX_LINES).min(total),
    };
    let actual_start = slice_start.min(total);

    let mut shortened = false;
    let selected: Vec<String> = lines[actual_start..slice_end.max(actual_start)]
        .iter()
        .map(|line| {
            if line.chars().count() > MAX_LINE_LENGTH {
                shortened = true;
                let head: String = line.chars().take(MAX_LINE_LENGTH).collect();
                format!("{head}... [truncated]")
            } else {
                line.to_string()
            }
        })
        .collect();
    let content = selected.join("\n");

    if actual_start > 0 || slice_end < total || shortened {
        format!(
            "\nIMPORTANT: The file content has been truncated.\n\
             Status: Showing lines {}-{slice_end} of {total} total lines.\n\
             Action: To read more of the file, you can use the 'start_line' and 'end_line' parameters in a subsequent 'read_file' call. For example, to read the next section of the file, use start_line: {}.\n\
             --- FILE CONTENT (truncated) ---\n\
             {content}",
            actual_start + 1,
            slice_end + 1,
        )
    } else {
        content
    }
}

/// `read_file` after Gemini CLI's (its Gemini 3 declaration): the raw text, no
/// line numbers. See [`super::get_read_tool_desc`] for Claude Code's.
pub fn get_read_file_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("read_file")
        .description(concat!(
            "Reads and returns the content of a specified file. ",
            "To maintain context efficiency, you MUST use 'start_line' and 'end_line' for targeted, surgical reads of specific sections. ",
            "For your safety, the tool will automatically truncate output exceeding 2000 lines, 2000 characters per line, or 10MB in size; ",
            "however, triggering these limits is considered token-inefficient. ",
            "Always retrieve only the minimum content necessary for your next step. ",
            "Handles text files; use `imgread` for images.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "file_path": {
                    "description": "The path to the file to read.",
                    "type": "string",
                },
                "start_line": {
                    "description": "Optional: The 1-based line number to start reading from.",
                    "type": "integer",
                    "minimum": 1,
                },
                "end_line": {
                    "description": "Optional: The 1-based line number to end reading at (inclusive).",
                    "type": "integer",
                    "minimum": 1,
                }
            },
            "required": ["file_path"]
        }))
        .build()
}

pub fn get_read_file_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, console: &mut ConsoleClient| -> Message {
            let path = args
                .pointer("/file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if path.trim().is_empty() {
                return error_message(id, "The 'file_path' parameter must be non-empty.");
            }
            let line = |key: &str| {
                args.pointer(key)
                    .and_then(|v| v.as_integer())
                    .map(|n| n.max(1) as usize)
            };
            let (start_line, end_line) = (line("/start_line"), line("/end_line"));
            if let (Some(start), Some(end)) = (start_line, end_line)
                && start > end
            {
                return error_message(id, "start_line cannot be greater than end_line");
            }
            let text = match load_text(console, path).await {
                Ok(text) => text,
                Err((e, _)) => return error_message(id, e),
            };
            let output = format_text(&text, start_line, end_line);
            Message::new(Role::Tool)
                .with_contents([Part::value(crate::to_value!({ "output": output.as_str() }))])
                .with_id(id)
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_returns_raw_text() {
        assert_eq!(format_text("alpha\nbeta", None, None), "alpha\nbeta");
        assert_eq!(format_text("a\r\nb", None, None), "a\nb");
    }

    #[test]
    fn test_read_file_notes_partial_range() {
        let out = format_text("a\nb\nc\nd\ne", Some(2), Some(3));
        assert!(
            out.contains("Status: Showing lines 2-3 of 5 total lines."),
            "{out}"
        );
        assert!(out.contains("use start_line: 4."), "{out}");
        assert!(
            out.ends_with("--- FILE CONTENT (truncated) ---\nb\nc"),
            "{out}"
        );
    }

    #[test]
    fn test_read_file_shortens_long_lines() {
        let long = "x".repeat(MAX_LINE_LENGTH + 5);
        let out = format_text(&long, None, None);
        assert!(out.contains("IMPORTANT: The file content has been truncated."));
        assert!(out.ends_with("... [truncated]"), "{out}");
    }
}
