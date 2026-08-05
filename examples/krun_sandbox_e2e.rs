//! End-to-end: ailoy's raw-msb_krun sandbox mounting a cortex S3 VFS.
//!
//! Proves the separation — cortex provides the filesystem, ailoy boots the VM
//! and mounts it — and the ephemeral model (exec capture + persistent upper).
//!
//! ```text
//! set -a; . ../agent-k/.env; set +a
//! export S3_BENCH_KEY=FinanceBench/3M_2023Q2_10Q.pdf
//! cargo run --example krun_sandbox_e2e --features sandbox
//! ```
//! The kernel and rootfs are resolved/provisioned by `Sandbox`. No manual
//! codesign: `Sandbox::exec` boots an ad-hoc-signed copy of this binary on macOS.

use ailoy::cortex::{S3Config, VolumeSpec, WorkspaceSpec};
use ailoy::runenv::Sandbox;

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("missing env {k}"))
}

fn main() {
    let upper = std::env::temp_dir().join("ailoy_krun_e2e.img");
    let _ = std::fs::remove_file(&upper);

    // A cortex workspace with one S3 mount at its root, described declaratively —
    // ailoy just carries the spec across to the VM helper.
    let volume = VolumeSpec::S3(S3Config {
        bucket: env("AWS_S3_BUCKET"),
        region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: env("AWS_ACCESS_KEY_ID"),
        secret_access_key: env("AWS_SECRET_ACCESS_KEY"),
        endpoint: std::env::var("AWS_S3_ENDPOINT").ok(),
        key_prefix: std::env::var("AWS_S3_KEY_PREFIX").ok(),
    });
    let workspace = WorkspaceSpec::default().mount("", volume);

    // ailoy resolves the kernel and provisions the base (Alpine) rootfs itself.
    // The whole workspace mounts at `/workspace` as one virtio-fs device.
    let sb = Sandbox::new(&upper)
        .expect("build sandbox")
        .with_workspace("/workspace", workspace);

    let r1 = sb.exec("echo hello; uname -sm", None).expect("exec1");
    println!("[1 capture] rc={}\n{}\n", r1.exit_code, r1.stdout);

    let key = std::env::var("S3_BENCH_KEY").unwrap_or_default();
    let r2 = sb
        .exec(
            &format!(
                "echo '[ls /workspace]'; ls /workspace; echo '[read S3 object]'; wc -c /workspace/{key}"
            ),
            None,
        )
        .expect("exec2");
    println!("[2 cortex S3 VFS] rc={}\n{}\n", r2.exit_code, r2.stdout);

    sb.exec("echo persist-9 > /data/m", None)
        .expect("exec3 write");
    let r4 = sb.exec("cat /data/m", None).expect("exec4 read");
    println!("[4 upper persistence, fresh vm] rc={}\n{}\n", r4.exit_code, r4.stdout);

    let ok_vfs = r2.stdout.contains("/workspace/");
    let ok_persist = r4.stdout.contains("persist-9");
    println!(
        "VFS mount: {} | persistence: {}",
        if ok_vfs { "PASS" } else { "FAIL" },
        if ok_persist { "PASS" } else { "FAIL" }
    );
}
