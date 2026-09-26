//! Speak a text in a voice described in words with Qwen3-TTS, through ncnn on the guest's Vulkan
//! device, as an agent's skill.
//!
//! ```sh
//! cargo run --example tts
//! ```
//!
//! [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B VoiceDesign speaks ten languages,
//! Korean among them, in a voice an instruction describes, such as "a calm woman in her thirties,
//! speaking slowly". It is three models, all of them here, converted from the checkpoint by
//! `prepare_model.py`: the talker, a Qwen3 LM that reads the instruction and the text and makes
//! each frame's first code; the code predictor, which makes the frame's other 15; and the codec's
//! decoder, which turns the frames into a 24 kHz waveform.
//!
//! There is no prompt. What to say and how are two files in the context folder, `text.txt` and
//! `instruct.txt`, and the agent reads them, speaks the text in that voice and hands back the
//! WAV.
//!
//! The skill is `SKILL.md` and `run_tts.py`, mounted at `/skills/tts` from memory, which the
//! agent runs with its `shell` tool.
//!
//! * `context/` at `/context`, read-only — the text and the instruction. When it is empty or
//!   missing, `context_example/` is copied into it first.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! The image is Debian rather than Alpine because PyPI's ncnn wheels are manylinux (glibc)
//! only. `mesa-vulkan-drivers` carries the venus ICD the guest needs — from trixie-backports,
//! as the other ncnn examples have it — and `libvulkan1` the loader the wheel opens.
//!
//! Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.
//!
//! Environment:
//!
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//!   It has to be built with the `gpu` feature, or the session is refused with
//!   `UNSUPPORTED_MACHINE`.
//! * `UV` — the `uv` binary `prepare_model.py` runs with, `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//!
//! Read from `.env` as well.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
use cortex::{fs::Directory, image::Recipe, protocol::NetworkAccess};
// One host binding per platform, each mounting on `try_new` and unmounting on `Drop`, so
// the tree below is written once. Three arms and not `not(windows)` because the guards are
// three distinct types: cortex's default `mount` feature compiles the one binding its target
// has — Dokany on Windows, `fuser` on Linux, FUSE-T on macOS — and names the guard after it.
#[cfg(windows)]
use cortex::fs::DokanMount as HostMount;
#[cfg(target_os = "linux")]
use cortex::fs::FuseMount as HostMount;
#[cfg(target_os = "macos")]
use cortex::fs::FuseTMount as HostMount;
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/tts");
    prepare(&project_path).await?;
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    // Something to say, for a first run.
    let context = project_path.join("context");
    if std::fs::read_dir(&context)?.next().is_none() {
        for entry in std::fs::read_dir(project_path.join("context_example"))? {
            let entry = entry?;
            std::fs::copy(entry.path(), context.join(entry.file_name()))
                .with_context(|| "copying context_example into context")?;
        }
    }

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: `/context`\n\n",
        "This folder holds the data and context the user wants to share with you. ",
        "When the user refers to something whose context you cannot figure out, the files in this folder might help. ",
        "The information that settles the answer may be here too, and so may hints toward it, so look through this folder for them.\n\n",
        "# Artifacts\n\n",
        "Path: `/artifacts`\n\n",
        "This folder is where what the user asked for goes. ",
        "Write every result here, such as a report, a figure, or the file the user came for. ",
        "Everything in this folder is collected and handed back to the user, and a result left anywhere else is not delivered.",
    ))
    .system_tools()
    .console(
        ConsoleClient::builder()
            .cmd(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(
                Recipe::new("python:3.12-slim-trixie")
                    // Mesa from backports, 26.0 against trixie's 25.0, as the other ncnn examples
                    // have it.
                    .step(
                        "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                        > /etc/apt/sources.list.d/backports.list \
                        && apt-get update && apt-get install -y --no-install-recommends \
                        -t trixie-backports mesa-vulkan-drivers \
                        && apt-get install -y --no-install-recommends libvulkan1 \
                        && rm -rf /var/lib/apt/lists/*",
                    )
                    .step("pip install --no-cache-dir ncnn numpy tokenizers"),
            )
            .mount_readonly(project_path.join("data/ncnn"), "/models")
            .mount_readonly(
                HostMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("run_tts.py", include_str!("run_tts.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/tts",
            )
            .mount_readonly(context, "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            // The build's `apt-get` and `pip` run with the session's reach.
            .network(NetworkAccess::public())
            .gpu(true)
            .vcpus(2)
            // The talker is 2.8 GB on the device in fp16, and its cache and the codec's buffers
            // come on top of it there; the run itself takes under 1 GB of memory.
            .memory_mib(4096)
            .gpu_memory_mib(8192)
            .build()
            .await
            .with_context(|| format!("starting the console"))?,
    )
    .skill("/skills/tts")
    .build()
    .await?;

    let query = Message::new(Role::User).with_contents([Part::text(
        "Speak the text in /context in the voice and manner its instruction describes, \
         and hand back the speech as a WAV file.",
    )]);
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
                        println!("→ {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            // A run prints where the speech went and how long it is, and ncnn's device log
            // and its progress: shown whole.
            Role::Tool => {
                for part in &message.contents {
                    println!("← {}", serde_json::to_string_pretty(part)?);
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }

    Ok(())
}

/// Download and convert the models into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the models: {status}");
    Ok(())
}
