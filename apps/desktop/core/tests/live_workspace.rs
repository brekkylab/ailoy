//! Needs FUSE-T installed. Run: `cargo test -p ailoy-desktop-core --test live_workspace -- --ignored`

use ailoy_desktop_core::workspace::{WorkspaceManager, fsops, is_mounted};

#[tokio::test]
#[ignore]
async fn a_mounted_workspace_is_visible_to_the_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let files = dir.path().join("files");
    let mp = dir.path().join("workspace");
    let ws = WorkspaceManager::start(
        files.clone(),
        dir.path().join("artifacts"),
        mp.clone(),
        true,
    )
    .await;
    assert!(
        matches!(
            ws.info().status,
            ailoy_desktop_core::WorkspaceStatus::Mounted
        ),
        "{:?}",
        ws.info()
    );
    assert!(is_mounted(&mp));

    fsops::write(&ws.fs(), "/via-workfs.txt", "kernel sees me")
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(mp.join("via-workfs.txt")).unwrap(),
        "kernel sees me"
    );
    assert_eq!(
        std::fs::read_to_string(files.join("via-workfs.txt")).unwrap(),
        "kernel sees me"
    );

    ws.shutdown().await;
    assert!(!is_mounted(&mp));
}
