//! Answer typed questions with Laya through ncnn on the guest's Vulkan device.
//!
//! ```sh
//! cargo run --example laya
//! ```
//!
//! [Laya](https://huggingface.co/convaiinnovations/laya) is a decision model: given a state
//! (a text, an email, a ticket) and typed questions — a choice, a score, a yes/no — it answers
//! each with calibrated probabilities in one forward pass, and generates no text.
//!
//! Two steps:
//!
//! 1. **Prepare** the model under `examples/laya/data/`, on the host, with `prepare_model.py`
//!    in the uv project in this directory: it downloads the English checkpoint (ModernBERT-large
//!    and Laya's decision head, ~840 MB, pinned to one revision) into `data/checkpoint/`, and
//!    converts it to ncnn into `data/ncnn/`, with the README's example request and laya's own
//!    answers to it from PyTorch. What it takes to get Laya through pnnx is in that file. A
//!    minute or two of CPU, done once.
//! 2. **Run** the request in fp32 and in bf16, in a GPU session with `data/ncnn` mounted
//!    read-only at `/models`: tokenized in the guest, one question an extraction, each answer
//!    checked against laya's. bf16 takes cortex-krun's bfloat16 patches to MoltenVK and
//!    SPIRV-Cross on the host; without them ncnn quietly runs fp32 instead, which the `bf16-p/s`
//!    line on stderr says. fp16 is not a mode: see `run_laya.py` for why.
//!
//! The image is Debian rather than Alpine because PyPI's ncnn wheels are manylinux
//! (glibc) only. `mesa-vulkan-drivers` carries the venus ICD the guest needs — from
//! trixie-backports, for bf16 — and `libvulkan1` the loader the wheel opens.
//!
//! Environment:
//!
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//!   It has to be built with the `gpu` feature, or the session is refused with
//!   `UNSUPPORTED_MACHINE`.
//! * `UV` — the `uv` binary `prepare_model.py` runs with, `uv` on `PATH` by default.
//!
//! Read from `.env` as well.

use std::path::Path;

use anyhow::Context as _;
use cortex::{
    console::{Console, NetworkAccess},
    image::Image,
};

const BASE: &str = "python:3.12-slim-trixie";
const RUN: &str = include_str!("run_laya.py");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| "cortex-krun".to_string());

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/laya");
    prepare(&project).await?;

    let mut console = Console::builder()
        .stdio_client(&[&program])
        .image(
            Image::new()
                .base(BASE)
                // Mesa from backports: venus passes VK_KHR_shader_bfloat16 through from 26.0
                // on, and trixie itself has 25.0.
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
        .mount_readonly(project.join("data/ncnn"), "/models")
        // The build's `apt-get` and `pip` run with the session's reach.
        .network(NetworkAccess::public())
        .gpu(true)
        .vcpus(2)
        .memory_mib(4096)
        .build()
        .await
        .with_context(|| format!("starting the console `{program}`"))?;

    let mut code = 0;
    for mode in ["fp32", "bf16"] {
        println!("=== laya, {mode} ===");
        let out = console
            .exec(["python3", "-c", RUN, mode], Some(600_000))
            .await?;
        print!("{}", String::from_utf8_lossy(&out.stdout));
        // ncnn logs the device it picked to stderr, which is part of the answer.
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        println!("--- exit {}\n", out.code);
        code = out.code;
        if code != 0 {
            break;
        }
    }

    // The last step's exit code is the verdict: 2 no Vulkan in the wheel, 3 no device
    // found, 4 tokens other than laya's, 5 an answer non-finite or too far from laya's.
    std::process::exit(code);
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
