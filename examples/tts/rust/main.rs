//! Speak a text in a voice described in words with Qwen3-TTS, through ncnn on the guest's Vulkan
//! device, as an agent's skill.
//!
//! ```sh
//! cargo run --example tts
//! ```
//!
//! [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B VoiceDesign speaks ten languages,
//! Korean among them, in a voice an instruction describes, such as "a calm woman in her thirties,
//! speaking slowly". It is three models, all converted from the checkpoint by `prepare_model.py`:
//! the talker, a Qwen3 LM that reads the instruction and the text and makes each frame's first
//! code; the code predictor, which makes the frame's other 15; and the codec's decoder, which
//! turns the frames into a 24 kHz waveform.
//!
//! There is no prompt: the agent reads `text.txt` and `instruct.txt` from the context folder,
//! speaks the text in that voice and hands back the WAV.
//!
//! The skill (`SKILL.md`, `run_tts.py`) is mounted from memory at `/skills/tts`.
//!
//! * `context/` at `/context`, read-only — the text and the instruction.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.
//!
//! Environment, also read from `.env`:
//!
//! * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
// One host binding per platform (mounts on `try_new`, unmounts on `Drop`), so the tree
// below is written once. Three arms, not `not(windows)`: virtx's default `mount` feature
// compiles only its target's binding, each a distinct guard type.
use futures::StreamExt as _;
#[cfg(windows)]
use virtx::fs::DokanMount as HostMount;
#[cfg(target_os = "linux")]
use virtx::fs::FuseMount as HostMount;
#[cfg(target_os = "macos")]
use virtx::fs::FuseTMount as HostMount;
use virtx::{fs::Directory, image::Recipe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Absolute, because a mount is named to the server as a `file://` URL.
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("tts/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("tts/rust");
    prepare(&shared_path, &project_path).await?;
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    // Something to say, for a first run.
    let context = project_path.join("context");
    if std::fs::read_dir(&context)?.next().is_none() {
        for entry in std::fs::read_dir(shared_path.join("context_example"))? {
            let entry = entry?;
            std::fs::copy(entry.path(), context.join(entry.file_name()))
                .with_context(|| "copying context_example into context")?;
        }
    }

    // The console server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none.
    virtx::ensure_virtx().await?;
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
    .max_tokens(64000)
    .system_tools()
    .console(
        ConsoleClient::builder()
            // Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
            .image(
                Recipe::new("python:3.12-slim-trixie")
                    // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader
                    // the wheel opens. Mesa from backports: 26.0 against trixie's 25.0.
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
                        .with_file("SKILL.md", include_str!("../shared/SKILL.md").as_bytes())?
                        .with_file("run_tts.py", include_str!("../shared/run_tts.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/tts",
            )
            .mount_readonly(context, "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
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
async fn prepare(shared: &Path, project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .arg(project.join("data"))
        .current_dir(shared)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the models: {status}");
    Ok(())
}
