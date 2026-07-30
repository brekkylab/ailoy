//! End-to-end: ailoy's raw-msb_krun sandbox mounting a cortex S3 VFS.
//!
//! Proves the separation — cortex provides the filesystem, ailoy boots the VM
//! and mounts it — and the ephemeral model (exec capture + persistent upper).
//!
//! ```text
//! set -a; . ../agent-k/.env; set +a
//! export CORTEX_TEST_KERNEL=$HOME/.microsandbox/lib/libkrunfw.dylib
//! export CORTEX_TEST_ROOTFS=../cortex/data/rootfs
//! export S3_BENCH_KEY=FinanceBench/3M_2023Q2_10Q.pdf
//! cargo run --example krun_sandbox_e2e --features krun   # then codesign + run
//! ```

use std::path::PathBuf;

use ailoy::runenv::{S3Vfs, Sandbox, boot_if_requested};

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("missing env {k}"))
}

fn main() {
    // MUST be first: a re-invoked child boots the VM here and never returns.
    boot_if_requested();

    let rootfs = std::env::var("CORTEX_TEST_ROOTFS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../cortex/data/rootfs"));
    let kernel = std::env::var("CORTEX_TEST_KERNEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".microsandbox/lib/libkrunfw.dylib")
        });
    let upper = std::env::temp_dir().join("ailoy_krun_e2e.img");
    let _ = std::fs::remove_file(&upper);

    let s3 = S3Vfs {
        bucket: env("AWS_S3_BUCKET"),
        region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: env("AWS_ACCESS_KEY_ID"),
        secret_access_key: env("AWS_SECRET_ACCESS_KEY"),
        endpoint: std::env::var("AWS_S3_ENDPOINT").ok(),
        key_prefix: std::env::var("AWS_S3_KEY_PREFIX").ok(),
        guest_path: "/workspace".to_string(),
    };

    let sb = Sandbox::new(&rootfs, &upper, &kernel)
        .expect("build sandbox")
        .with_s3(s3);

    let r1 = sb.exec("echo hello; uname -sm").expect("exec1");
    println!("[1 capture] rc={}\n{}\n", r1.exit_code, r1.stdout);

    let key = std::env::var("S3_BENCH_KEY").unwrap_or_default();
    let r2 = sb
        .exec(&format!(
            "echo '[ls /workspace]'; ls /workspace; echo '[read S3 object]'; wc -c /workspace/{key}"
        ))
        .expect("exec2");
    println!("[2 cortex S3 VFS] rc={}\n{}\n", r2.exit_code, r2.stdout);

    sb.exec("echo persist-9 > /data/m").expect("exec3 write");
    let r4 = sb.exec("cat /data/m").expect("exec4 read");
    println!("[4 upper persistence, fresh vm] rc={}\n{}\n", r4.exit_code, r4.stdout);

    let ok_vfs = r2.stdout.contains("/workspace/");
    let ok_persist = r4.stdout.contains("persist-9");
    println!(
        "VFS mount: {} | persistence: {}",
        if ok_vfs { "PASS" } else { "FAIL" },
        if ok_persist { "PASS" } else { "FAIL" }
    );
}
