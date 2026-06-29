//! Host FUSE frontend tests: the non-sandbox local mount and an interactive
//! host-mount inspector.

use crate::common::*;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS + ANTHROPIC creds + macFUSE host"]
async fn e2e_non_sandbox_host_fuser() {
    dotenvy::dotenv().ok();
    let s = stamp();
    let fname = format!("e2e-nosandbox-{s}.txt");
    let content = format!("nosandbox-{s}");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .vfs(all_vfs())
        .build()
        .expect("build agent");
    let transcript = drive(agent, &task_for(&fname, &content)).await;
    println!(
        "--- non-sandbox transcript tail ---\n{}",
        tail(&transcript, 500)
    );
    assert!(
        verify(&format!("/s3/{fname}"), &content).await,
        "non-sandbox write not found in S3"
    );
}

/// Non-sandbox counterpart to `vfs_inspect_sandbox`: builds a host-FUSE agent
/// (no sandbox) with S3 + Notion + GDrive mounted at a host temp dir, runs it
/// once, then sleeps so you can inspect the mount directly from any terminal:
///   ls -la <printed mount dir>
///   cat <printed mount dir>/notion/pages/<page>/page.json
/// Requires macFUSE/libfuse on the host.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "interactive: holds a host FUSE mount up for manual inspection (needs macFUSE)"]
async fn vfs_inspect_host() {
    dotenvy::dotenv().ok();

    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool.")
        .shell_tool()
        .vfs(all_vfs())
        .build()
        .expect("build agent (host FUSE mount happens here)");

    // Run once to confirm the host shell can traverse the mount. The host mount
    // is already up at build time; keep `agent` alive so its `AgentVfs` (and the
    // FUSE mount) is not dropped/unmounted while we sleep.
    // {
    //     let q = Message::new(Role::User)
    //         .with_contents([Part::text("Reply with the single word READY.")]);
    //     let mut strm = agent.run(q);
    //     while let Some(ev) = strm.next().await {
    //         if let Err(e) = ev {
    //             eprintln!("run error (mount may have failed): {e}");
    //         }
    //     }
    // }

    // The builder mounts at std::env::temp_dir()/ailoy-vfs-<pid>-<n>; discover it.
    let prefix = format!("ailoy-vfs-{}-", std::process::id());
    let mount_dir = std::fs::read_dir(std::env::temp_dir()).ok().and_then(|rd| {
        rd.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix))
                .unwrap_or(false)
        })
    });
    let mount = mount_dir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("{}/{}<n>", std::env::temp_dir().display(), prefix));

    println!("\n================ VFS INSPECT (HOST FUSE) ================");
    println!("host mount   : {mount}");
    println!("mounts       : {mount}/{{s3,notion,gdrive}}");
    println!("inspect from another terminal (plain host shell, no sandbox):");
    println!("  mount | grep -i fuse");
    println!("  ls -la {mount}");
    println!("  ls {mount}/s3");
    println!("  ls {mount}/notion/pages");
    println!("  ls {mount}/gdrive | head");
    println!("sleeping 1h. Ctrl-C this process when done, then unmount:");
    println!("  umount {mount}   # or: diskutil unmount {mount}");
    println!("========================================================\n");

    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    drop(agent);
}
