//! Answer typed questions with Laya through ncnn on the guest's Vulkan device, alone or as an
//! agent's tool.
//!
//! ```sh
//! cargo run --example laya
//! cargo run --example laya -- "Triage this ticket: we were billed twice for March …"
//! ```
//!
//! [Laya](https://huggingface.co/convaiinnovations/laya) is a decision model: given a state
//! (a text, an email, a ticket) and typed questions — a choice, a score, a yes/no — it answers
//! each with calibrated probabilities in one forward pass, and generates no text.
//!
//! * `context/` at `/context`, read-only — what to decide on, when it is not in the prompt.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! The image is Alpine: ncnn, numpy and tokenizers all ship musllinux wheels.
//! `mesa-vulkan-virtio` carries the venus ICD the guest needs — Mesa 26.1 on 3.24, for
//! bf16 — and `vulkan-loader` the loader the wheel opens.
//!
//! Environment:
//!
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//!   It has to be built with the `gpu` feature, or the session is refused with
//!   `UNSUPPORTED_MACHINE`.
//! * `UV` — the `uv` binary `prepare_model.py` runs with, `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//! * `LAYA_MODE` — what the `laya` tool runs in, `fp32` (the default) or `bf16`.
//!
//! Read from `.env` as well.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::Console,
    datatype::Value,
    message::{Message, Part, Role},
    to_value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc, get_tool_providers_mut},
    tool_func,
};
use anyhow::Context as _;
use cortex::{console::NetworkAccess, image::Image};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| "cortex-krun".to_string());

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/laya");
    prepare(&project_path).await?;
    for dir in ["context", "artifacts"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    let mode = std::env::var("LAYA_MODE").unwrap_or_else(|_| "fp32".to_string());
    anyhow::ensure!(
        ["fp32", "bf16"].contains(&mode.as_str()),
        "LAYA_MODE is fp32 or bf16, not {mode}"
    );

    get_tool_providers_mut()
        .get_mut("default")
        .context("no default tool provider")?
        .insert_func("laya", laya_tool_func(mode.clone()));

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: /context\n\n",
        "Holds what you were given to decide on, such as the user's messages, ",
        "tickets and documents. When the request refers to something that is not in it, look ",
        "here first. List the folder, read what bears on the request, and pass that text to Laya ",
        "as the state. This folder is read-only.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the files the user asks for go. When they ask for a file, such as ",
        "a report or a table of decisions, write it here and name it so they can tell what it ",
        "is. A result that is only in your reply is not delivered as a file.",
    ))
    .system_tools()
    .tool(laya_tool_desc())
    .console(
        Console::builder()
            .stdio_client(&[&program])
            .image(
                Image::new()
                    .base("python:3.12-alpine3.24")
                    // Alpine 3.24 ships Mesa 26.1: venus passes VK_KHR_shader_bfloat16 through from
                    // 26.0 on.
                    .step("apk add --no-cache vulkan-loader mesa-vulkan-virtio")
                    .step("pip install --no-cache-dir ncnn numpy tokenizers"),
            )
            .mount_readonly(project_path.join("data/ncnn"), "/models")
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            // The build's `apk` and `pip` run with the session's reach.
            .network(NetworkAccess::public())
            .gpu(true)
            .vcpus(2)
            .memory_mib(4096)
            .build()
            .await
            .with_context(|| format!("starting the console `{program}`"))?,
    )
    .build()?;

    let query = Message::new(Role::User).with_contents([Part::text(prompt)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let message = output?.message;
        match message.role {
            Role::Assistant => {
                for text in message.contents.iter().filter_map(Part::as_text) {
                    println!("{text}");
                }
                for call in message.tool_calls.iter().flatten() {
                    if let Some((_, name, args)) = call.as_function() {
                        println!("  → {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            // Laya's answers are small, and they are what the reply is made from: shown whole.
            Role::Tool => {
                for part in &message.contents {
                    println!("  ← {}", serde_json::to_string_pretty(part)?);
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }

    Ok(())
}

/// Download and convert the model into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        // `transformers` probes for TensorFlow at import, which can hang model construction.
        .env("USE_TF", "0")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the model: {status}");
    Ok(())
}

fn laya_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("laya")
        .description(
            "Laya is a machine learning model for decisions. Given a text, such as a message, \
            an email or a ticket, it answers typed questions about it with calibrated \
            probabilities. It picks one of several options, rates on an ordered scale, or \
            answers yes or no, and it generates no text.",
        )
        .parameters(to_value!({
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The text the questions are about, passed verbatim."
                },
                "questions": {
                    "type": "object",
                    "description": "The questions, keyed by an id of your choosing such as department or urgency.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["choice", "score", "noul"],
                                "description": "A choice picks one of the criteria and gives a probability for each. A score picks a level on an ordered scale and is answered as its expected value. A noul is a yes or no question and is answered as the probability of yes."
                            },
                            "instructions": {
                                "type": "string",
                                "description": "The question in one sentence."
                            },
                            "criteria": {
                                "description": "For a choice, the options as a list of labels or as an object from each label to what it covers. For a score, the levels as a list of descriptions from lowest to highest. For a noul, optionally an object that says what true and false mean."
                            }
                        },
                        "required": ["type", "instructions"]
                    }
                }
            },
            "required": ["state", "questions"]
        }))
        .build()
}

/// `run_laya.py` on the call's request, in the console the agent was given.
fn laya_tool_func(mode: String) -> ToolFunc {
    tool_func!(async |args: Value, console: &mut Console| -> Value
        with [mode = mode.clone()]
        {
            let request = match serde_json::to_string(&args) {
                Ok(request) => request,
                Err(e) => return to_value!({ "error": format!("encoding the request: {e}") }),
            };
            match console
                .exec(["python3", "-c", include_str!("run_laya.py"), &mode, &request], Some(600_000))
                .await
            {
                Ok(out) if out.code == 0 => {
                    match serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                        Ok(answers) => Value::from(answers),
                        Err(e) => to_value!({ "error": format!("reading laya's answers: {e}") }),
                    }
                }
                Ok(out) => to_value!({
                    "error": format!(
                        "laya exited {}: {}",
                        out.code,
                        String::from_utf8_lossy(&out.stderr).trim()
                    )
                }),
                Err(e) => to_value!({ "error": format!("running laya: {e}") }),
            }
        }
    )
}
