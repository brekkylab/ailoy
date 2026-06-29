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

/// Host FUSE frontend write model + dir ops via real `std::fs` syscalls on the
/// mounted dir (no agent): full write, append (C1), truncate (C2), rename and
/// remove (C3). Exercises the host fuse.rs callbacks directly — the forwarder
/// tests cover the sandbox path, this covers the host path.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: host FUSE write/append/truncate/rename/rm (needs AWS + macFUSE)"]
async fn vfs_host_write_ops() {
    use std::{io::Write, sync::Arc};
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap());
    let rt = tokio::runtime::Handle::current();
    let mp = std::env::temp_dir().join(format!("ailoy-hostw-{}", stamp()));
    std::fs::create_dir_all(&mp).unwrap();
    let _mount = ailoy::vfs::VfsMount::spawn(vfs, &mp, rt).expect("host fuse mount");

    // Wait for the mount to be live (s3 dir readable).
    for _ in 0..40 {
        if std::fs::read_dir(mp.join("s3")).is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    let s = stamp();
    let f = mp.join("s3").join(format!("hostw-{s}.txt"));
    let g = mp.join("s3").join(format!("hostw-{s}-moved.txt"));

    // Full write, then append must preserve it (C1), then truncate keeps the head (C2).
    std::fs::write(&f, b"AAAA").unwrap();
    assert_eq!(std::fs::read(&f).unwrap(), b"AAAA", "full write");
    {
        let mut h = std::fs::OpenOptions::new().append(true).open(&f).unwrap();
        h.write_all(b"BBBB").unwrap();
    }
    assert_eq!(
        std::fs::read(&f).unwrap(),
        b"AAAABBBB",
        "append must preserve existing (C1)"
    );
    {
        let h = std::fs::OpenOptions::new().write(true).open(&f).unwrap();
        h.set_len(3).unwrap();
    }
    assert_eq!(
        std::fs::read(&f).unwrap(),
        b"AAA",
        "truncate must keep the head (C2)"
    );

    // Rename (C3): destination has the content, source is gone.
    std::fs::rename(&f, &g).unwrap();
    assert_eq!(std::fs::read(&g).unwrap(), b"AAA", "rename moves content");
    assert!(std::fs::read(&f).is_err(), "rename leaves no source");

    // Remove (C3).
    std::fs::remove_file(&g).unwrap();
    assert!(std::fs::read(&g).is_err(), "file still present after rm");
}
