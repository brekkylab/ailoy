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
//! Four volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-tts-models` at `/models`, read-only — `data/ncnn/` beside this file, the converted
//!   models, put there each run.
//! * `ailoy-tts-skill` at `/skills/tts`, read-only — the skill (`SKILL.md`, `run_tts.py`) from
//!   `../shared/`, put there each run.
//! * `ailoy-tts-context` at `/context`, read-only — `context/` beside this file, the text and the
//!   instruction, put there each run.
//! * `ailoy-tts-artifacts` at `/artifacts`, writable — where what the agent hands back goes. It
//!   stays on the server between runs, and is copied to `artifacts/` beside this file at the end.
//!
//! Qwen3-TTS and its weights are the Qwen team's, under Apache-2.0.
//!
//! Environment, also read from `.env`:
//!
//! * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//!   has to be set up as `docs/src/content/docs/guide/model-providers.mdx` says.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    message::{Message, Part, Role},
};
use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use virtx::{Client, Files as _, ImageJobEntry, Recipe, SandboxCreateArgs, SandboxVolume};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-tts:1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("tts/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("tts/rust");
    prepare(&shared_path, &project_path).await?;
    for dir in ["context", "artifacts"] {
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

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once. Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
    // The server answers a tag in its full form, as `docker.io/library/ailoy-tts:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        let recipe = Recipe::new("python:3.12-slim-trixie")
            .step("pip install --no-cache-dir ncnn numpy tokenizers");
        let mut log = client.build_image(recipe, IMAGE).log();
        while let Some(entry) = log.next().await {
            if let ImageJobEntry::End(end) = entry?.entry
                && let Some(error) = end.error
            {
                bail!("building {IMAGE}: {error}");
            }
        }
    }

    // The volumes, made the first time and kept on the server: `409` says one is there already.
    let mut volumes = Vec::new();
    for name in [
        "ailoy-tts-models",
        "ailoy-tts-skill",
        "ailoy-tts-context",
        "ailoy-tts-artifacts",
    ] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [models, skill, context_volume, artifacts] = <[_; 4]>::try_from(volumes).unwrap();

    // `data/ncnn/` and `context/`, every file of them, onto theirs.
    for (from, volume) in [
        (project_path.join("data/ncnn"), &models),
        (context, &context_volume),
    ] {
        let mut dirs = vec![(from, String::new())];
        while let Some((dir, at)) = dirs.pop() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let to = format!("{at}{}", entry.file_name().to_string_lossy());
                if entry.file_type()?.is_dir() {
                    volume.create_dir_all(to.clone()).await?;
                    dirs.push((entry.path(), format!("{to}/")));
                } else {
                    volume
                        .write_file(to, std::fs::read(entry.path())?, None)
                        .await?;
                }
            }
        }
    }
    // The skill, from `shared/`.
    for name in ["SKILL.md", "run_tts.py"] {
        skill
            .write_file(name, std::fs::read(shared_path.join(name))?, None)
            .await?;
    }

    let mut args = SandboxCreateArgs::new(IMAGE);
    args.volumes = vec![
        SandboxVolume {
            volume: models.name().to_string(),
            path: "/models".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: skill.name().to_string(),
            path: "/skills/tts".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: context_volume.name().to_string(),
            path: "/context".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: artifacts.name().to_string(),
            path: "/artifacts".to_string(),
            readonly: false,
        },
    ];
    args.gpu = Some(true);
    args.vcpus = Some(2);
    // The talker is 2.8 GB on the device in fp16, and its cache and the codec's buffers come on
    // top of it there; the run itself takes under 1 GB of memory.
    args.memory_mib = Some(4096);
    args.gpu_memory_mib = Some(8192);
    let sandbox = client.create_sandbox(&args).await?;

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
    .sandbox(sandbox)
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

    // The artifacts, every file of them, to `artifacts/` here.
    let mut dirs = vec![String::new()];
    while let Some(at) = dirs.pop() {
        std::fs::create_dir_all(project_path.join("artifacts").join(&at))?;
        for entry in artifacts.read_dir(at.clone()).await? {
            let from = format!("{at}{}", entry.name);
            if entry.metadata.is_dir {
                dirs.push(format!("{from}/"));
            } else {
                let bytes = artifacts.read_file(from.clone(), 0, None).await?;
                std::fs::write(project_path.join("artifacts").join(&from), bytes)?;
            }
        }
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
