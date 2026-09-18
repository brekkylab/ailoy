//! Needs a built `cortex-local-console` (AILOY_CORTEX_BIN_DIR or ../cortex/target/debug).
//! Run: `cargo test -p ailoy-desktop-core --test live_console -- --ignored`

use ailoy_desktop_core::{
    console::{ConsoleFactory, resolve_console_bin},
    workspace::WorkspaceMount,
};

/// The two trees a run is given, and which one a command starts in.
///
/// Cortex mounts a session's *context* read-only and starts it in its *scratch*, so the
/// workspace goes in as the session's artifacts: the agent has to be able to write there.
/// This pins both halves of that arrangement — a relative path lands in the scratch, and the
/// workspace is reachable and writable by its own path — because getting the roles the wrong
/// way round fails only at the first write, which is far from here.
#[tokio::test]
#[ignore]
async fn a_console_starts_in_its_scratch_and_can_write_the_workspace() {
    let workspace = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("marker.txt"), b"here").unwrap();

    let factory = ConsoleFactory::new(resolve_console_bin(None).unwrap());
    let mut console = factory
        .spawn(
            WorkspaceMount(workspace.path().to_path_buf()),
            WorkspaceMount(scratch.path().to_path_buf()),
        )
        .await
        .unwrap();

    // Where it stands: a relative path is the scratch, not the user's files.
    let out = console.exec(["pwd"], Some(5_000)).await.unwrap();
    let cwd = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        std::fs::canonicalize(&cwd).unwrap() == std::fs::canonicalize(scratch.path()).unwrap(),
        "session stands in {cwd}, expected the scratch at {}",
        scratch.path().display()
    );

    // The workspace is there, by its own path.
    let marker = workspace.path().join("marker.txt");
    let out = console
        .exec(["cat", &marker.display().to_string()], Some(5_000))
        .await
        .unwrap();
    assert_eq!(
        out.code,
        0,
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"here".to_vec());

    // And writable — the whole reason it is the artifacts tree rather than the context.
    let written = workspace.path().join("written-by-the-agent.txt");
    console
        .write(
            written.display().to_string(),
            b"from the session".to_vec(),
            None,
        )
        .await
        .expect("the workspace takes a write");
    assert_eq!(
        std::fs::read_to_string(&written).unwrap(),
        "from the session"
    );
}
