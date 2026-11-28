//! The `browser` abin of the sandbox, virtx's Chromium driven by agent-browser, as two tools:
//! `browser` runs one of its commands, and `browser_screenshot` answers with what the page
//! looks like as an image. The abin's `SKILL.md`, listed in the system message, says what the
//! commands are.

use super::{exec_result, image_mime};
use crate::{
    datatype::Bytes,
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

/// The abin these tools run, by the name virtx gives it in a sandbox.
pub const BROWSER_ABIN: &str = "browser";

/// How long one command may take: a page that never settles should not hold the turn.
const TIMEOUT_MS: u64 = 120_000;

fn error_message(id: String, msg: impl Into<String>) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::value(crate::to_value!({ "error": msg.into() }))])
        .with_id(id)
}

pub fn get_browser_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("browser")
        .description(concat!(
            "Run one command of the sandbox's `browser` (agent-browser driving Chromium), ",
            "as its skill describes: `[\"open\", \"https://example.com\"]`, ",
            "`[\"snapshot\", \"-i\"]`, `[\"click\", \"@e3\"]`, `[\"fill\", \"@e2\", \"text\"]`. ",
            "Every command uses the same browser, with its tabs, cookies and refs. ",
            "Take a new snapshot after anything changes the page.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The command and its arguments, each one string, without `browser` itself"
                }
            },
            "required": ["args"]
        }))
        .build()
}

pub fn get_browser_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, sandbox: &Sandbox| -> Value {
        let Some(items) = args.pointer("/args").and_then(|v| v.as_array()) else {
            return crate::to_value!({
                "stdout": "",
                "stderr": "missing required parameter: args",
                "exit_code": -1,
                "phase": "validation"
            });
        };
        let mut argv = vec![BROWSER_ABIN.to_owned()];
        for item in items {
            match item.as_str() {
                Some(arg) => argv.push(arg.to_owned()),
                None => {
                    return crate::to_value!({
                        "stdout": "",
                        "stderr": "every one of args has to be a string",
                        "exit_code": -1,
                        "phase": "validation"
                    });
                }
            }
        }
        exec_result(sandbox, argv, Some(TIMEOUT_MS)).await
    })
}

pub fn get_browser_screenshot_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("browser_screenshot")
        .description(concat!(
            "See the browser's current page as an image. ",
            "Prefer `browser snapshot`, which is smaller and says what can be clicked; ",
            "take a screenshot for what only shows when seen: layout, images, charts.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "The whole height of the page rather than what is in view"
                },
                "annotate": {
                    "type": "boolean",
                    "description": "Label each element with its ref, as `browser snapshot` names them"
                }
            }
        }))
        .build()
}

pub fn get_browser_screenshot_tool_func() -> ToolFunc {
    tool_func!(
        async |args: Value, id: String, sandbox: &Sandbox| -> Message {
            let path = format!("/tmp/ailoy-screenshot-{}.png", uuid::Uuid::new_v4());
            let mut argv = vec![BROWSER_ABIN.to_owned(), "screenshot".into(), path.clone()];
            for flag in ["full", "annotate"] {
                if args.pointer(&format!("/{flag}")).and_then(|v| v.as_bool()) == Some(true) {
                    argv.push(format!("--{flag}"));
                }
            }

            let out = match crate::console::exec(sandbox, argv, Some(TIMEOUT_MS)).await {
                Ok(out) if out.timed_out => return error_message(id, "the screenshot timed out"),
                Ok(out) if out.code != 0 => {
                    let why = if out.stderr.trim().is_empty() {
                        out.stdout
                    } else {
                        out.stderr
                    };
                    return error_message(id, why.trim().to_owned());
                }
                Ok(out) => out,
                Err(e) => return error_message(id, e.to_string()),
            };
            let read = crate::console::read(sandbox, &path).await;
            // Left behind, it would fill the sandbox's disk over a long run.
            let _ = crate::console::exec(sandbox, ["rm", "-f", path.as_str()], None).await;
            let bytes = match read {
                Ok(bytes) => bytes,
                Err(e) => return error_message(id, format!("reading the screenshot: {e}")),
            };
            let Some(mime) = image_mime(&bytes) else {
                return error_message(id, "the screenshot is not an image");
            };

            let image = Part::image_embedded(mime, Bytes::from(bytes))
                .expect("image_embedded always succeeds");
            let mut contents = vec![image];
            // `--annotate` prints the legend of the labels.
            if !out.stdout.trim().is_empty() {
                contents.push(Part::text(out.stdout.trim()));
            }
            Message::new(Role::Tool).with_contents(contents).with_id(id)
        }
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{console::Sandbox, test_sandbox, to_value};

    /// A 1x1 PNG.
    const PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// A sandbox whose `browser` prints its arguments, one per line, and for `screenshot`
    /// writes [`PNG`] where it is told and says so: the tools are checked against what they
    /// run, not against Chromium.
    async fn sandbox_with_fake_browser() -> Sandbox {
        let sandbox = test_sandbox().await;
        crate::console::write(&sandbox, "/tmp/shot.png", PNG)
            .await
            .unwrap();
        let script = concat!(
            "#!/bin/sh\n",
            "if [ \"$1\" = screenshot ]; then cp /tmp/shot.png \"$2\"; echo legend; exit 0; fi\n",
            "if [ \"$1\" = fail ]; then echo no page >&2; exit 1; fi\n",
            "for a in \"$@\"; do echo \"$a\"; done\n",
        );
        crate::console::write(&sandbox, "/usr/local/bin/browser", script)
            .await
            .unwrap();
        crate::console::exec(&sandbox, ["chmod", "+x", "/usr/local/bin/browser"], None)
            .await
            .unwrap();
        sandbox
    }

    #[tokio::test]
    async fn test_browser_passes_each_arg_as_it_is() {
        let sandbox = sandbox_with_fake_browser().await;
        let args = to_value!({ "args": ["fill", "@e2", "two words; $HOME"] });
        let out = get_browser_tool_func()
            .call(args, "1", &sandbox)
            .next()
            .await
            .unwrap()
            .message;
        let value = out.contents[0].as_value().unwrap();
        assert_eq!(
            value.pointer("/stdout").and_then(|v| v.as_str()),
            Some("fill\n@e2\ntwo words; $HOME\n")
        );
        assert_eq!(
            value.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_browser_without_args_is_refused() {
        let sandbox = test_sandbox().await;
        let out = get_browser_tool_func()
            .call(to_value!({}), "1", &sandbox)
            .next()
            .await
            .unwrap()
            .message;
        let value = out.contents[0].as_value().unwrap();
        assert_eq!(
            value.pointer("/phase").and_then(|v| v.as_str()),
            Some("validation")
        );
    }

    #[tokio::test]
    async fn test_screenshot_is_an_image_with_its_legend() {
        let sandbox = sandbox_with_fake_browser().await;
        let args = to_value!({ "annotate": true });
        let out = get_browser_screenshot_tool_func()
            .call(args, "1", &sandbox)
            .next()
            .await
            .unwrap()
            .message;
        assert!(out.contents[0].is_image(), "{out:?}");
        assert_eq!(out.contents[1].as_text(), Some("legend"));
        let left = crate::console::exec(&sandbox, ["sh", "-c", "ls /tmp/ailoy-screenshot-*"], None)
            .await
            .unwrap();
        assert_ne!(left.code, 0, "the screenshot is removed: {}", left.stdout);
    }
}
