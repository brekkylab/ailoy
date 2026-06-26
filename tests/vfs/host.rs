//! Host FUSE frontend tests: the non-sandbox local mount, the host-FUSE
//! into-guest virtiofs probe, and an interactive host-mount inspector.

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

/// PROBE the "fundamentally different" design: mount the provider FUSE on the
/// HOST (macFUSE), then bind that host dir into the guest as a microsandbox
/// volume (virtiofs). If the guest can read provider files through the bind,
/// this needs NO in-guest process and microsandbox re-applies the bind on every
/// VM start — surviving restarts for free. Risk: virtiofs may refuse to export
/// a FUSE mountpoint ("Operation not permitted").
#[tokio::test(flavor = "multi_thread")]
#[ignore = "probe: host-FUSE bound into guest via virtiofs (needs creds + macFUSE + sandbox)"]
async fn vfs_host_fuse_bind_into_guest() {
    use std::sync::Arc;

    dotenvy::dotenv().ok();
    // Host FUSE mount of the s3 provider, with our own fixture file to look for
    // (no assumption that any specific provider content already exists).
    let fx = Fixture::create("/s3").await;
    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap());
    let rt = tokio::runtime::Handle::current();
    let mountpoint = std::env::temp_dir().join(format!("ailoy-hostfuse-{}", stamp()));
    std::fs::create_dir_all(&mountpoint).unwrap();
    let _mount = ailoy::vfs::VfsMount::spawn(vfs, &mountpoint, rt).expect("host fuse mount");

    // Wait for the host FUSE mount to be ready + show our fixture (first readdir
    // triggers the provider fetch) before binding it into the guest.
    let mut host_sees = false;
    for _ in 0..40 {
        host_sees = std::fs::read_dir(mountpoint.join("s3"))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy() == fx.name())
            })
            .unwrap_or(false);
        if host_sees {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    println!("host sees fixture {}: {host_sees}", fx.name());
    assert!(host_sees, "host FUSE mount not ready / fixture not visible");

    let name = format!("ailoy-hostbind-{}", stamp());
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: false,
        allow_host_egress: true,
        volumes: vec![ailoy::runenv::VolumeMount::Bind {
            host: mountpoint.clone(),
            guest: "/mnt/vfs".into(),
            readonly: false,
        }],
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let h = sandbox.get().await.expect("handle");
    for _ in 0..40 {
        if h.exec_shell("true".into(), Some(10))
            .await
            .map(|o| o.exit_code == 0)
            .unwrap_or(false)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    let out = h
        .exec_shell(
            "echo '== ls /mnt/vfs =='; ls /mnt/vfs 2>&1; \
             echo '== ls /mnt/vfs/s3 =='; ls /mnt/vfs/s3 2>&1; \
             echo '== mount =='; mount | grep -i 'mnt/vfs\\|virtiofs' 2>&1"
                .into(),
            Some(30),
        )
        .await
        .expect("guest ls");
    println!(
        "GUEST:\n{}\n--- stderr ---\n{} (exit={}, timed_out={})",
        out.stdout, out.stderr, out.exit_code, out.timed_out
    );
    fx.teardown().await;
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

    let mut agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool.")
        .shell_tool()
        .vfs(all_vfs())
        .build()
        .expect("build agent (host FUSE mount happens here)");

    // Run once to confirm the host shell can traverse the mount. The host mount
    // is already up at build time; keep `agent` alive so its `AgentVfs` (and the
    // FUSE mount) is not dropped/unmounted while we sleep.
    {
        let q = Message::new(Role::User)
            .with_contents([Part::text("Reply with the single word READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            if let Err(e) = ev {
                eprintln!("run error (mount may have failed): {e}");
            }
        }
    }

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
}
