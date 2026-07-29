//! End-to-end: mount a cortex-backed S3 source into a real microsandbox
//! sandbox via the fs-backend path, and read it from inside the guest.
//!
//! Exercises the whole integration at runtime:
//!   ailoy SandboxBuilder.mount(FsBackend)
//!     -> SDK SandboxConfig::add_fs_backend -> LaunchConfig.fs_backends
//!     -> `$MSB_PATH sandbox` (= cortex's `msb_cortex`, which registered the
//!        "cortex-s3" factory) -> build_vm resolves it -> virtio-fs device
//!     -> agentd auto-mounts it (MSB_DIR_MOUNTS) at the guest path
//!     -> guest `ls`/`wc` reads S3 (fetched host-side by the msb_cortex process).
//!
//! Run (S3 creds + the codesigned msb_cortex binary):
//! ```text
//! set -a; . ../agent-k/.env; set +a
//! export MSB_PATH=$PWD/../cortex/target/debug/msb_cortex
//! cargo run --example cortex_s3_e2e --features sandbox
//! ```

use ailoy::runenv::{Console as _, Machine as _, SandboxBuilder, VolumeMount};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::var("MSB_PATH").is_err() {
        anyhow::bail!("set MSB_PATH to the codesigned cortex `msb_cortex` binary");
    }
    let var = |k: &str| std::env::var(k).map_err(|_| anyhow::anyhow!("missing env {k}"));

    // The `params` blob cortex's "cortex-s3" factory deserializes (its field
    // names match agent-k's workspace `S3Config`).
    let params = serde_json::json!({
        "bucket": var("AWS_S3_BUCKET")?,
        "region": std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        "access_key_id": var("AWS_ACCESS_KEY_ID")?,
        "secret_access_key": var("AWS_SECRET_ACCESS_KEY")?,
        "endpoint": std::env::var("AWS_S3_ENDPOINT").ok(),
        "key_prefix": std::env::var("AWS_S3_KEY_PREFIX").ok(),
    })
    .to_string();

    let guest = "/mnt/workspace/s3";
    println!("building sandbox (alpine) with cortex-s3 fs-backend at {guest} ...");
    let mut sandbox = SandboxBuilder::new()
        .image("alpine:latest")
        .mount(VolumeMount::FsBackend {
            tag: "wsfs0".to_string(),
            guest: guest.to_string(),
            backend_type: "cortex-s3".to_string(),
            params,
        })
        .build()
        .await?;

    let console = sandbox.start().await?;
    println!("sandbox started; reading the auto-mounted S3 source in the guest ...");

    // Prove the mount exists and is populated from S3. If S3_BENCH_KEY is set,
    // also size that specific object through the mount.
    let key = std::env::var("S3_BENCH_KEY").unwrap_or_default();
    let script = format!(
        "echo '[mount]'; grep wsfs0 /proc/mounts || true; \
         echo '[ls {guest}]'; ls -la {guest} 2>&1 | head -20; \
         if [ -n '{key}' ]; then echo '[wc {key}]'; wc -c {guest}/{key} 2>&1; fi"
    );
    let res = console.exec_shell(script, Some(60)).await?;
    println!("--- exit={} ---\n{}", res.exit_code, res.stdout);
    if !res.stderr.is_empty() {
        println!("[stderr]\n{}", res.stderr);
    }

    sandbox.stop().await.ok();
    Ok(())
}
