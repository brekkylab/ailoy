//! Run one agent turn over a context mount and an artifacts mount.
//!
//! ```sh
//! cargo run --example run -- "Summarize what is in the context"
//! ```
//!
//! The session is given two mounts, both taken from one data directory on the host:
//!
//! * `<data>/context` at `/context`, read-only — what the agent is given to work from.
//! * `<data>/artifacts` at `/artifacts`, writable — where what it hands back goes.
//!
//! `<data>` is this example's own directory, `examples/run/`, unless `AILOY_DATA`
//! names another one.
//! Both subdirectories are created if missing, so a first run works on an empty context.
//!
//! The agent has the filesystem tools (`read`, `write`, `edit`, `glob`, `grep` —
//! `apply_patch` in place of `write`/`edit` on `openai/*`) and the network
//! tools (`web_search`, `web_fetch`). Not `shell`: every tool it has goes through a
//! named file or a URL.
//!
//! Environment:
//!
//! * `AILOY_MODEL` — defaults to `anthropic/claude-sonnet-5`; its provider's API key
//!   has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default
//!   (built in the sibling `cortex-krun` checkout). It runs the session in a micro-VM,
//!   which is what puts the two mounts at `/context` and `/artifacts`.
//! * `AILOY_DATA` — the data directory described above.
//!
//! * `AILOY_ROOTFS` — the image the micro-VM boots, `alpine:3.21` by default.
//!
//! All four are read from `.env` as well.

use std::{io::Write as _, path::PathBuf};

use ailoy::{
    agent::{Agent, AgentSpec, AgentState},
    message::{Message, Part, Role},
};
use anyhow::{Context as _, bail};
use cortex::{console::Console, image::Image};
use futures::StreamExt as _;

const CONTEXT_AT: &str = "/context";
const ARTIFACTS_AT: &str = "/artifacts";

/// What the mounts are *for*. The runtime appends where they are, but not what each
/// one means — that is this caller's to say.
fn instruction() -> String {
    format!(
        concat!(
            "You are a research assistant working on the user's request with files and the web.\n\n",
            "# Context — `{context}` (read-only)\n\n",
            "What you were given to work from: the user's own documents and data, and the",
            " background you need before starting. Look at what is here first — list it,",
            " then read what bears on the request — and prefer it over a guess or a web",
            " search whenever both could answer. You cannot write here.\n\n",
            "# Artifacts — `{artifacts}` (writable)\n\n",
            "Where what the user asked for goes. Every deliverable — a report, a table, the",
            " file they came for — is written here, because the contents of this directory",
            " are what is handed back; a result left only in your reply is not delivered.",
            " Put finished work here, named so the user can tell what each file is.\n\n",
        ),
        context = CONTEXT_AT,
        artifacts = ARTIFACTS_AT,
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ailoy loads `.env` only under `#[cfg(test)]`, so a binary has to.
    dotenvy::dotenv().ok();

    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if query.trim().is_empty() {
        bail!("usage: cargo run --example run -- <request>");
    }

    let model =
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string());
    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| "cortex-krun".to_string());
    // `cortex-krun` boots a micro-VM, and refuses a session that names no base to boot.
    let rootfs = std::env::var("AILOY_ROOTFS").unwrap_or_else(|_| "alpine:3.21".to_string());
    let data = std::env::var("AILOY_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/run"));

    // A mount travels as a `file://` URL, which needs an absolute path.
    let context = data.join("context");
    let artifacts = data.join("artifacts");
    for dir in [&context, &artifacts] {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let context = std::fs::canonicalize(context)?;
    let artifacts = std::fs::canonicalize(artifacts)?;

    let console = Console::builder()
        .stdio_client(&[&program])
        .image(Image::new().base(&rootfs))
        .mount_readonly(context.clone(), CONTEXT_AT)
        .mount(artifacts.clone(), ARTIFACTS_AT)
        .build()
        .await
        .with_context(|| format!("starting the console `{program}`"))?;

    let mut spec = AgentSpec::new(&model)
        .instruction(instruction())
        .system_tools()
        .web_search_tool(vec![])
        .web_fetch_tool()
        // A deliverable is written through a tool call, so its whole body counts
        // against one response; the provider default is too small for a real report.
        .max_tokens(32_000);
    // Filesystem tools only: `system_tools` brings `shell` along with them.
    spec.tools.retain(|t| t.name != "shell");

    let mut agent = Agent::try_with_provider_and_state(
        spec,
        "default",
        AgentState::new().with_console(console),
    )?;

    println!("  model      {model}");
    println!("  context    {} -> {CONTEXT_AT} (ro)", context.display());
    println!(
        "  artifacts  {} -> {ARTIFACTS_AT} (rw)\n",
        artifacts.display()
    );

    let query = Message::new(Role::User).with_contents([Part::text(query)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let output = output?;
        let message = &output.message;
        match message.role {
            Role::Assistant => {
                for text in message.contents.iter().filter_map(Part::as_text) {
                    println!("{text}");
                }
                for call in message.tool_calls.iter().flatten() {
                    if let Some((_, name, args)) = call.as_function() {
                        println!("  → {name} {}", serde_json::to_string(args)?);
                    }
                }
            }
            // A tool result is usually a value part rather than text, so it is shown
            // as JSON, cut to one line.
            Role::Tool => {
                let result = serde_json::to_string(&message.contents)?;
                let preview: String = result.chars().take(160).collect();
                let more = if preview.len() < result.len() {
                    "…"
                } else {
                    ""
                };
                println!("  ← {preview}{more}");
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }
    drop(stream);

    println!("\n--- artifacts ---");
    for entry in std::fs::read_dir(&artifacts)? {
        let entry = entry?;
        println!(
            "  {:>8}  {}",
            entry.metadata()?.len(),
            entry.file_name().to_string_lossy()
        );
    }
    Ok(())
}
