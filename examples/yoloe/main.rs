//! Detect objects named in plain text with YOLOE, through ncnn on the guest's Vulkan device.
//!
//! ```sh
//! cargo run --example yoloe
//! YOLOE_CLASSES="person,bus,glasses" YOLOE_IMAGE=street.jpg cargo run --example yoloe
//! ```
//!
//! [YOLOE](https://docs.ultralytics.com/models/yoloe/) is an open-vocabulary detector: it looks
//! for whatever classes it is given as text, rather than the 80 COCO classes a YOLO is trained
//! on. The text goes through MobileCLIP once, on the host, into embeddings YOLOE folds into its
//! head, so the model in the guest is as fast as a plain YOLO and finds those classes.
//!
//! Two steps:
//!
//! 1. **Prepare** the model under `examples/yoloe/data/`, on the host, with `prepare_model.py`
//!    in the uv project in this directory: it downloads YOLOE-11l-seg and MobileCLIP (~600 MB)
//!    into `data/weights/`, sets the classes, and converts it to ncnn into
//!    `data/ncnn/` with Ultralytics' own export, with the image and what Ultralytics finds in it
//!    with the PyTorch model. Done again only when the model, the classes or the image change.
//! 2. **Run** the detection in fp32, fp16 and bf16, in a GPU session with `data/ncnn` mounted
//!    read-only at `/models`, each checked against Ultralytics'. bf16 takes cortex-krun's
//!    bfloat16 patches to MoltenVK and SPIRV-Cross on the host; without them ncnn quietly runs
//!    fp32 instead, which the `bf16-p/s` line on stderr says.
//!
//! The image is Debian rather than Alpine because PyPI's ncnn wheels are manylinux (glibc)
//! only. `mesa-vulkan-drivers` carries the venus ICD the guest needs — from trixie-backports,
//! for bf16 — and `libvulkan1` the loader the wheel opens. The ncnn wheel goes in without its
//! dependencies, which would bring in `opencv-python` and the X and GL libraries it needs; the
//! headless OpenCV does the letterboxing instead.
//!
//! YOLOE and its weights are Ultralytics', under AGPL-3.0.
//!
//! Environment:
//!
//! * `YOLOE_CLASSES` — the classes to look for, comma-separated; see `prepare_model.py` for
//!   the default.
//! * `YOLOE_IMAGE` — the image to look in, Ultralytics' `bus.jpg` by default.
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
const RUN: &str = include_str!("run_yoloe.py");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| "cortex-krun".to_string());

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yoloe");
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
                .step(
                    "pip install --no-cache-dir --no-deps ncnn \
                    && pip install --no-cache-dir numpy opencv-python-headless",
                ),
        )
        .mount_readonly(project.join("data/ncnn"), "/models")
        // The build's `apt-get` and `pip` run with the session's reach.
        .network(NetworkAccess::public())
        .gpu(true)
        .vcpus(2)
        .memory_mib(2048)
        .build()
        .await
        .with_context(|| format!("starting the console `{program}`"))?;

    // A process each, as for SAM3: ncnn does not take a net in one precision after one in
    // another in the same process.
    let mut code = 0;
    for mode in ["fp32", "fp16", "bf16"] {
        println!("=== yoloe, {mode} ===");
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
    // found, 5 a non-finite output or detections other than Ultralytics'.
    std::process::exit(code);
}

/// Download, set the classes of and convert the model into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let mut command = tokio::process::Command::new(&uv);
    // Named from where this runs, not from the project directory `uv` runs in.
    if let Some(image) = std::env::var_os("YOLOE_IMAGE") {
        let image = std::path::absolute(&image)
            .with_context(|| format!("resolving YOLOE_IMAGE {}", image.display()))?;
        command.env("YOLOE_IMAGE", image);
    }
    let status = command
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the model: {status}");
    Ok(())
}
