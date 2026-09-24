//! Run the SAM3 encoders through ncnn on the guest's Vulkan device.
//!
//! ```sh
//! cargo run --example sam3
//! ```
//!
//! Two steps:
//!
//! 1. **Prepare** the models under `examples/sam3/data/`, on the host, with
//!    `prepare_model.py` in the uv project in this directory: it downloads the ONNX export
//!    of the SAM3 encoders (~3.4 GB, pinned to one revision) into `data/onnx/` and converts
//!    them to ncnn into `data/ncnn/`. What it takes to get SAM3 through pnnx is in that file.
//!    ~7 GB at its peak and a minute or two of CPU, done once.
//! 2. **Run** both encoders in fp32 and in bf16, in a GPU session with `data/ncnn` mounted
//!    read-only at `/models`. bf16 is checked against fp32 by cosine similarity. It takes
//!    cortex-krun's bfloat16 and cooperative matrix patches to MoltenVK and SPIRV-Cross on
//!    the host; without them ncnn quietly runs fp32 instead, which the `bf16-p/s` and
//!    `bf16-cm` lines on stderr say.
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
const RUN: &str = include_str!("run_encoders.py");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| "cortex-krun".to_string());

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/sam3");
    prepare(&project).await?;

    let mut console = Console::builder()
        .stdio_client(&[&program])
        .image(
            Image::new()
                .base(BASE)
                // Mesa from backports: venus passes VK_KHR_shader_bfloat16 and
                // VK_KHR_cooperative_matrix through from 26.0 on, and trixie itself has 25.0.
                .step(
                    "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                    > /etc/apt/sources.list.d/backports.list \
                    && apt-get update && apt-get install -y --no-install-recommends \
                    -t trixie-backports mesa-vulkan-drivers \
                    && apt-get install -y --no-install-recommends libvulkan1 \
                    && rm -rf /var/lib/apt/lists/*",
                )
                .step("pip install --no-cache-dir ncnn numpy"),
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

    // Each model in fp32 and then in bf16, which is checked against it. A process each:
    // see `run_encoders.py` for why.
    let (lang, img) = ("sam3_language_encoder", "sam3_image_encoder");
    let steps: [(&str, Vec<&str>); 4] = [
        ("language encoder, fp32", vec![lang, "fp32"]),
        ("language encoder, bf16", vec![lang, "bf16", "fp32"]),
        ("image encoder, fp32", vec![img, "fp32"]),
        ("image encoder, bf16", vec![img, "bf16", "fp32"]),
    ];
    let mut code = 0;
    for (label, args) in steps {
        println!("=== {label} ===");
        let cmd = [vec!["python3", "-c", RUN], args].concat();
        let out = console.exec(cmd, Some(600_000)).await?;
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
    // found, 5 a non-finite output, or one too far from fp32's.
    std::process::exit(code);
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
