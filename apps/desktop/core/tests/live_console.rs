//! Needs a built `cortex-local-console` (AILOY_CORTEX_BIN_DIR or ../cortex/target/debug).
//! Run: `cargo test -p ailoy-desktop-core --test live_console -- --ignored`

use ailoy_desktop_core::{
    console::{ConsoleFactory, resolve_console_bin},
    workspace::WorkspaceMount,
};

#[tokio::test]
#[ignore]
async fn a_console_stands_in_the_mount_it_was_given() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), b"here").unwrap();
    let factory = ConsoleFactory::new(resolve_console_bin(None).unwrap());
    let mut console = factory
        .spawn(WorkspaceMount(dir.path().to_path_buf()))
        .await
        .unwrap();
    let out = console
        .exec(["cat", "marker.txt"], Some(5_000))
        .await
        .unwrap();
    assert_eq!(
        out.code,
        0,
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"here".to_vec());
}
