//! Answer typed questions with Laya through ncnn on the guest's Vulkan device, as an agent's
//! skill.
//!
//! ```sh
//! cargo run --example laya
//! cargo run --example laya -- "Route the tickets in /context/inbox, but send anything about contracts to legal"
//! ```
//!
//! [Laya](https://huggingface.co/convaiinnovations/laya) is a decision model: given a state
//! (a text, an email, a ticket) and typed questions — a choice, a score, a yes/no — it answers
//! each with calibrated probabilities in one forward pass, and generates no text.
//!
//! Without a prompt the agent routes a support inbox: it asks Laya which department each ticket in
//! `shared/tickets/` belongs to and files a copy in that department's folder.
//!
//! Four volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-laya-models` at `/models`, read-only — `data/ncnn/`, the converted model, put there
//!   each run.
//! * `ailoy-laya-skill` at `/skills/laya`, read-only — the skill, `SKILL.md` and `run_laya.py`
//!   from `../shared/`, put there each run.
//! * `ailoy-laya-context` at `/context`, read-only — `context/`, what to decide on, when it is
//!   not in the prompt, put there each run. `context/inbox/` is a fresh copy of
//!   `shared/tickets/` on every run.
//! * `ailoy-laya-artifacts` at `/artifacts`, writable — where what the agent hands back goes: the
//!   tickets filed under `routes/<department>/`, emptied at the start of every run. It stays on
//!   the server between runs, and is copied to `artifacts/` beside this file at the end.
//!
//! Environment, also read from `.env`:
//!
//! * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//!   has to be set up as `docs/content/guides/model-providers.mdx` says.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    message::{Message, Part, Role},
};
use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use virtx::{Client, Files as _, ImageJobEntry, Recipe, SandboxCreateArgs, SandboxVolume};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-laya:1";

/// The request when none is given.
const QUERY: &str = "Route every ticket in /context/inbox to the department that should handle it, \
    by what the customer needs rather than the words they use: engineering, finance, sales, legal \
    or marketing. Decide each one with Laya, copy the ticket into /artifacts/routes/<department>/, \
    and write what went where to /artifacts/routing.md.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("laya/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("laya/rust");
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        QUERY.to_string()
    } else {
        prompt
    };

    prepare(&shared_path, &project_path).await?;

    for dir in ["context", "artifacts"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    fill_inbox(&shared_path, &project_path)?;
    clear_routes(&project_path)?;

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once. The server answers a tag in its full form, as
    // `docker.io/library/ailoy-laya:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        // Debian, not Alpine: PyPI's musllinux ncnn wheel crashes freeing its first `Mat`,
        // where the manylinux (glibc) one runs laya.
        let recipe = Recipe::new("python:3.12-slim-trixie")
            // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader
            // the wheel opens. Mesa from backports: venus passes VK_KHR_shader_bfloat16
            // through from 26.0 on, and trixie itself has 25.0.
            .step(
                "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                > /etc/apt/sources.list.d/backports.list \
                && apt-get update && apt-get install -y --no-install-recommends \
                -t trixie-backports mesa-vulkan-drivers \
                && apt-get install -y --no-install-recommends libvulkan1 \
                && rm -rf /var/lib/apt/lists/*",
            )
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
        "ailoy-laya-models",
        "ailoy-laya-skill",
        "ailoy-laya-context",
        "ailoy-laya-artifacts",
    ] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [models, skill, context, artifacts] = <[_; 4]>::try_from(volumes).unwrap();

    // The skill, `SKILL.md` and `run_laya.py` of `shared/`, onto its own.
    for name in ["SKILL.md", "run_laya.py"] {
        skill
            .write_file(name, std::fs::read(shared_path.join(name))?, None)
            .await?;
    }
    // A fresh inbox and no routes, as on this side.
    context.remove_file("inbox", true).await?;
    artifacts.remove_file("routes", true).await?;
    // `data/ncnn/` and `context/`, every file of them, onto theirs.
    for (volume, root) in [
        (&models, project_path.join("data/ncnn")),
        (&context, project_path.join("context")),
    ] {
        let mut dirs = vec![(root, String::new())];
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

    let mut args = SandboxCreateArgs::new(IMAGE);
    args.volumes = vec![
        SandboxVolume {
            volume: models.name().to_string(),
            path: "/models".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: skill.name().to_string(),
            path: "/skills/laya".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: context.name().to_string(),
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
    args.memory_mib = Some(4096);
    args.gpu_memory_mib = Some(12288);
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
    .web_fetch_tool()
    .web_search_tool(vec![])
    .sandbox(sandbox)
    .skill("/skills/laya")
    .build()
    .await?;

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
                        println!("→ {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            // Laya's answers are small, and they are what the reply is made from: shown whole.
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

/// Reset `project/context/inbox` to a copy of `shared/tickets`, so every run routes the same inbox.
fn fill_inbox(shared: &Path, project: &Path) -> anyhow::Result<()> {
    let inbox = project.join("context/inbox");
    if inbox.exists() {
        std::fs::remove_dir_all(&inbox).with_context(|| format!("clearing {}", inbox.display()))?;
    }
    std::fs::create_dir_all(&inbox).with_context(|| format!("creating {}", inbox.display()))?;
    let tickets = shared.join("tickets");
    for entry in
        std::fs::read_dir(&tickets).with_context(|| format!("reading {}", tickets.display()))?
    {
        let path = entry?.path();
        if path.is_file() {
            std::fs::copy(&path, inbox.join(path.file_name().unwrap()))
                .with_context(|| format!("copying {}", path.display()))?;
        }
    }
    Ok(())
}

/// Empty `project/artifacts/routes`, so what is filed there is only this run's.
fn clear_routes(project: &Path) -> anyhow::Result<()> {
    let routes = project.join("artifacts/routes");
    if routes.exists() {
        std::fs::remove_dir_all(&routes)
            .with_context(|| format!("clearing {}", routes.display()))?;
    }
    Ok(())
}

/// Download and convert the model into `project/data`
async fn prepare(shared: &Path, project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .arg(project.join("data"))
        .current_dir(shared)
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
