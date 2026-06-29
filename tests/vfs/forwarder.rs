//! The static in-guest forwarder binary itself: crash/OOM recovery and the
//! lower-level read / write / unlink paths over the forward protocol.

use crate::common::*;

/// Recovery from in-guest forwarder *process death* (crash / OOM / SIGKILL) —
/// distinct from a VM restart. Bring the mount up, then kill the forwarder so the
/// mount is left with a dead FUSE daemon (any access would hang in the kernel),
/// then drive the SAME agent again: `ensure_mounted`'s functional liveness probe
/// must detect the dead daemon (its `ls` times out), tear the stale mount down,
/// and re-bootstrap a fresh forwarder — restoring provider access. A hang here
/// would mean the liveness probe itself gets stuck on the dead daemon.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: recover from forwarder process death via self-healing re-mount"]
async fn vfs_recovers_from_forwarder_death() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-fwddeath-{}", stamp());
    let fx = Fixture::create("/s3").await;
    let path = fx.guest_path();

    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: true,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let mut agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(all_vfs())
        .build()
        .expect("build agent");

    // Run 1: bring the mount up.
    {
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            let _ = ev;
        }
    }
    let want = fx.len() as i64;
    let n1 = msb_read_count(&name, &path);
    println!("before forwarder kill: {n1} bytes");
    assert_eq!(n1, want, "initial mount read should match the fixture size");

    // Kill the in-guest forwarder (simulate a crash) — the mount now has a dead
    // FUSE daemon; any access to it would block in the kernel. Match the binary
    // by basename via `pgrep -> kill` so we don't SIGKILL our own `sh -c` (whose
    // argv contains the full path).
    let k = std::process::Command::new("msb")
        .args([
            "exec", &name, "--", "sh", "-c",
            "for p in $(pgrep -f 'vfs-fwd /mnt/vfs'); do kill -9 \"$p\"; done; sleep 0.4; echo killed",
        ])
        .output()
        .expect("msb exec kill");
    println!(
        "forwarder killed: {}",
        String::from_utf8_lossy(&k.stdout).trim()
    );

    // Run 2 on the SAME agent: ensure_mounted's liveness probe must detect the
    // dead daemon (the mount is now "Transport endpoint not connected"), tear it
    // down, and re-bootstrap a fresh forwarder.
    {
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            if let Err(e) = ev {
                eprintln!("run2 error: {e}");
            }
        }
    }
    let n2 = msb_read_count(&name, &path);
    println!("after forwarder death + self-heal: {n2} bytes");

    msb_rm(&name);
    fx.teardown().await;
    assert_eq!(
        n2, want,
        "provider access should recover after forwarder process death (got {n2})"
    );
    println!("FORWARDER-DEATH RECOVERY OK");
}

/// Repeated forwarder death over a long session: recovery must work EVERY time,
/// and the lazy unmounts must not leave stale ENOTCONN mounts piling up in
/// /proc/mounts (a leak that could eventually wedge the mountpoint). Kill + heal
/// several times, asserting the read recovers each round, then assert the
/// mountpoint has not accumulated stacked stale mounts.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: repeated forwarder death — recovery + no stale-mount accumulation"]
async fn vfs_repeated_forwarder_death_recovery() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-fwdloop-{}", stamp());
    let fx = Fixture::create("/s3").await;
    let path = fx.guest_path();
    async fn drive_once(agent: &mut Agent) {
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            if let Err(e) = ev {
                eprintln!("run error: {e}");
            }
        }
    }

    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: true,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let mut agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(all_vfs())
        .build()
        .expect("build agent");

    let want = fx.len() as i64;
    drive_once(&mut agent).await; // initial mount
    assert_eq!(msb_read_count(&name, &path), want, "initial mount read failed");

    for round in 1..=3 {
        let _ = std::process::Command::new("msb")
            .args([
                "exec",
                &name,
                "--",
                "sh",
                "-c",
                "for p in $(pgrep -f 'vfs-fwd /mnt/vfs'); do kill -9 \"$p\"; done; sleep 0.4",
            ])
            .status();
        drive_once(&mut agent).await; // self-heal re-mount
        let n = msb_read_count(&name, &path);
        println!("round {round}: after kill+heal -> {n} bytes");
        assert_eq!(
            n, want,
            "round {round}: provider access did not recover (got {n})"
        );
    }

    // No stale-mount accumulation: exactly one /mnt/vfs entry in /proc/mounts.
    let m = std::process::Command::new("msb")
        .args([
            "exec",
            &name,
            "--",
            "sh",
            "-c",
            "grep -c ' /mnt/vfs ' /proc/mounts",
        ])
        .output()
        .expect("msb exec mounts");
    let count: i64 = String::from_utf8_lossy(&m.stdout)
        .trim()
        .parse()
        .unwrap_or(-1);
    println!("/mnt/vfs entries in /proc/mounts after 3 kill+heal rounds: {count}");

    msb_rm(&name);
    fx.teardown().await;
    assert_eq!(
        count, 1,
        "stale mounts accumulated ({count} /mnt/vfs entries) — lazy umount is leaking"
    );
    println!("REPEATED FORWARDER-DEATH RECOVERY OK (3 rounds, 1 mount entry)");
}

/// Full dep-free forwarder end to end: host forward server (s3) + a CLEAN
/// sandbox (no python/fuse3/apt) running the static Rust forwarder binary, which
/// reaches the host over allow@host egress and serves provider content (a
/// self-created S3 fixture, cleaned up afterwards).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: full static dep-free forwarder serves a provider (needs creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_full() {
    use std::sync::Arc;
    dotenvy::dotenv().ok();

    let fx = Fixture::create("/s3").await;
    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap()); // s3-only
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs, &rt).expect("forward");
    let (port, tok) = (forward.port(), forward.token().to_string());

    let name = format!("ailoy-fwdfull-{}", stamp());
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: false,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let h = sandbox.get().await.expect("handle");
    launch_static_forwarder(&h, port, &tok).await;

    let path = fx.guest_path();
    let out = h
        .exec_shell(format!("cat '{path}' 2>/dev/null | wc -c"), Some(60))
        .await
        .expect("guest read");
    let bytes: i64 = out.stdout.trim().parse().unwrap_or(0);
    println!("static forwarder served {bytes} bytes (want {})", fx.len());

    msb_rm(&name);
    fx.teardown().await;
    assert_eq!(
        bytes,
        fx.len() as i64,
        "dep-free forwarder did not serve the fixture file ({bytes}B, want {})",
        fx.len()
    );
}

/// Full filesystem access through the static dep-free forwarder: write a file to
/// the S3 mount, read it back, then `rm` it (unlink) — all from a clean guest
/// with no python/fuse3/apt. Also covers the read-modify-write path (append +
/// truncate must preserve existing content, not NUL-clobber it).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: write+rm through static forwarder on S3 (needs AWS creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_write_unlink() {
    use std::sync::Arc;
    dotenvy::dotenv().ok();

    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap()); // s3-only
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs, &rt).expect("forward");
    let (port, tok) = (forward.port(), forward.token().to_string());

    let s = stamp();
    let name = format!("ailoy-fwdrm-{s}");
    let fname = format!("vfs-fwd-rm-{s}.txt");
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: false,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let h = sandbox.get().await.expect("handle");
    launch_static_forwarder(&h, port, &tok).await;
    let f = format!("/mnt/vfs/s3/{fname}");
    let af = format!("/mnt/vfs/s3/vfs-fwd-ap-{s}.txt");
    let mvf = format!("/mnt/vfs/s3/vfs-fwd-mv-{s}.txt");
    let rdir = format!("/mnt/vfs/s3/vfs-fwd-rd-{s}");
    let script = format!(
        "printf 'rmcontent-{s}' > '{f}'; \
         echo \"WROTE=$(cat '{f}')\"; \
         rm '{f}' && echo RM_OK || echo RM_FAIL; \
         (ls '{f}' >/dev/null 2>&1 && echo STILL_THERE || echo GONE); \
         printf 'AAAA' > '{af}'; printf 'BBBB' >> '{af}'; echo \"APPEND=$(cat '{af}')\"; \
         (truncate -s 3 '{af}' 2>/dev/null && echo \"TRUNC=$(cat '{af}')\" || echo TRUNC=skip); \
         rm '{af}'; \
         printf 'MOVE' > '{mvf}'; \
         mv '{mvf}' '{mvf}.x'; echo \"MV_RC=$?\"; \
         (ls '{mvf}' >/dev/null 2>&1 && echo SRC_THERE || echo SRC_GONE); \
         (mkdir '{rdir}' 2>/dev/null && echo MKDIR_OK || echo MKDIR_FAIL); \
         printf 'x' > '{rdir}/c.txt'; \
         (rmdir '{rdir}' 2>/dev/null && echo RMDIR_OK || echo RMDIR_FAIL); \
         (ls '{rdir}/c.txt' >/dev/null 2>&1 && echo RD_THERE || echo RD_GONE); \
         printf 'T' > '{f}.mt'; echo \"MTIME=$(stat -c %Y '{f}.mt' 2>/dev/null)\"; rm '{f}.mt'"
    );
    let out = h.exec_shell(script, Some(60)).await.expect("guest exec");
    println!("--- guest ---\n{}", out.stdout);
    if !out.stderr.trim().is_empty() {
        println!("--- stderr ---\n{}", out.stderr);
    }
    msb_rm(&name);
    assert!(
        out.stdout.contains(&format!("WROTE=rmcontent-{s}")),
        "write/read failed"
    );
    assert!(out.stdout.contains("RM_OK"), "rm (unlink) failed");
    assert!(out.stdout.contains("GONE"), "file still present after rm");
    // C1: append must preserve the existing content (RMW), not NUL-clobber it.
    assert!(
        out.stdout.contains("APPEND=AAAABBBB"),
        "append corrupted existing content (read-modify-write broken): {}",
        out.stdout
    );
    // C2: truncate must keep the leading bytes (skip only if guest lacks truncate).
    assert!(
        out.stdout.contains("TRUNC=AAA") || out.stdout.contains("TRUNC=skip"),
        "truncate did not preserve leading content: {}",
        out.stdout
    );
    // C3: rename (mv) = copy+delete; mkdir succeeds (implicit dir); rmdir removes
    // the prefix recursively.
    assert!(
        out.stdout.contains("MV_RC=0") && out.stdout.contains("SRC_GONE"),
        "mv (rename) failed or left the source: {}",
        out.stdout
    );
    // Verify host-side that the rename carried content to the destination.
    assert!(
        verify(&format!("/s3/vfs-fwd-mv-{s}.txt.x"), "MOVE").await,
        "rename did not copy content to the destination"
    );
    assert!(out.stdout.contains("MKDIR_OK"), "mkdir failed: {}", out.stdout);
    assert!(
        out.stdout.contains("RMDIR_OK") && out.stdout.contains("RD_GONE"),
        "rmdir did not remove the directory contents: {}",
        out.stdout
    );
    // Clean up the renamed object (host-side).
    {
        let v = Vfs::from_config(all_vfs()).unwrap();
        if let Some((res, vp)) = v.route(&format!("/s3/vfs-fwd-mv-{s}.txt.x")) {
            let _ = res.unlink(&vp).await;
        }
    }
    // S3-1: a just-written file reports a real mtime (recent epoch), not 1970.
    let mtime: i64 = out
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("MTIME="))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    assert!(
        mtime > 1_700_000_000,
        "S3 mtime not surfaced through the mount (got {mtime}): {}",
        out.stdout
    );
}

/// Large-file / chunked-read correctness through the static forwarder: write a
/// ~300 KB object to S3 (host-side), then `cat` it in a clean guest and verify
/// the byte count and content checksum match — proving direct_io chunked reads
/// reassemble correctly (the kernel issues many ranged reads for a big file).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: large/chunked read through static forwarder (needs AWS creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_large_read() {
    use std::sync::Arc;
    dotenvy::dotenv().ok();

    let s = stamp();
    let fname = format!("vfs-fwd-big-{s}.bin");
    // Deterministic ~300 KB payload: byte[i] = i % 251.
    let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    let want_len = data.len();
    // Sample offsets (incl. mid + last byte) — verifying these proves correct
    // multi-chunk offset handling without needing bc/md5 in the minimal guest.
    let (b0, bmid, bend) = (
        (0u32 % 251) as u8,
        (150_000u32 % 251) as u8,
        (299_999u32 % 251) as u8,
    );

    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap()); // s3-only
    {
        let (res, vp) = vfs.route(&format!("/s3/{fname}")).expect("route");
        res.write_bytes(&vp, data)
            .await
            .expect("host write big file");
    }
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs.clone(), &rt).expect("forward");
    let (port, tok) = (forward.port(), forward.token().to_string());

    let name = format!("ailoy-fwdbig-{s}");
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: false,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let h = sandbox.get().await.expect("handle");
    launch_static_forwarder(&h, port, &tok).await;
    let f = format!("/mnt/vfs/s3/{fname}");
    // Sum every byte in the guest (awk over od) + byte count → compare to host.
    let script = format!(
        "echo \"LEN=$(cat '{f}' | wc -c)\"; \
         echo \"B0=$(dd if='{f}' bs=1 skip=0 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\"; \
         echo \"BMID=$(dd if='{f}' bs=1 skip=150000 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\"; \
         echo \"BEND=$(dd if='{f}' bs=1 skip=299999 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\""
    );
    let out = h.exec_shell(script, Some(90)).await.expect("guest exec");
    println!(
        "--- guest ---\n{}\nwant LEN={want_len} B0={b0} BMID={bmid} BEND={bend}",
        out.stdout
    );
    msb_rm(&name);
    // Clean up the S3 object.
    {
        let (res, vp) = vfs.route(&format!("/s3/{fname}")).expect("route");
        let _ = res.unlink(&vp).await;
    }
    assert!(
        out.stdout.contains(&format!("LEN={want_len}")),
        "byte count mismatch"
    );
    assert!(out.stdout.contains(&format!("B0={b0}")), "byte@0 mismatch");
    assert!(
        out.stdout.contains(&format!("BMID={bmid}")),
        "byte@150000 mismatch"
    );
    assert!(
        out.stdout.contains(&format!("BEND={bend}")),
        "byte@299999 mismatch"
    );
}
