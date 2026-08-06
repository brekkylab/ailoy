//! End-to-end: pull a public OCI image, boot it as the sandbox rootfs, and run
//! Python inside — proving `Sandbox::with_image` yields a bootable guest.
//!
//! ```text
//! cargo run --example oci_boot_e2e --features sandbox
//! ```
//! The kernel and rootfs are resolved by `Sandbox`. No manual codesign:
//! `Sandbox::exec` boots an ad-hoc-signed copy on macOS.

use ailoy::cortex::{VolumeSpec, WorkspaceSpec};
use ailoy::runenv::Sandbox;

fn main() {
    let image = std::env::var("OCI_TEST_IMAGE").unwrap_or_else(|_| "python:3.12-slim".into());

    // A cortex Local workspace with a known file, to prove the *workspace*
    // virtio-fs share mounts inside a glibc image (the case util-linux `mount`
    // could not do — now handled by the guest init's mount(2)).
    let ws_host = std::env::temp_dir().join("ailoy_oci_ws");
    std::fs::create_dir_all(&ws_host).unwrap();
    std::fs::write(ws_host.join("hello.txt"), b"workspace-mount-ok\n").unwrap();
    let workspace = WorkspaceSpec::default().mount("", VolumeSpec::Local { host: ws_host });

    // `with_image` pulls/unpacks (async); `exec` boots the VM (sync). Drive the
    // pull on a throwaway runtime, then run.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let sandbox = rt.block_on(async {
        Sandbox::new()
            .expect("build sandbox")
            .with_image(&image)
            .await
            .expect("pull image rootfs")
            .with_workspace("/workspace", workspace)
    });

    let r = sandbox
        .exec(
            "echo '[uname]'; uname -sm; \
             echo '[python]'; python3 --version; \
             echo '[timeit]'; python3 -m timeit -n 1000 -r 3 '1+1'; \
             echo '[workspace]'; cat /workspace/hello.txt",
            None,
        )
        .expect("exec");
    println!("rc={}\n{}", r.exit_code, r.stdout);

    let py_ok = r.stdout.contains("Python 3.") && r.stdout.contains("per loop");
    let ws_ok = r.stdout.contains("workspace-mount-ok");
    println!(
        "python: {} | workspace mount: {}",
        if py_ok { "PASS" } else { "FAIL" },
        if ws_ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if py_ok && ws_ok { 0 } else { 1 });
}
