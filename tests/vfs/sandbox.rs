//! Sandbox frontend lifecycle: the in-guest forwarder mounted via the host
//! forward server, and its survival across VM restarts, reconnects, concurrency,
//! multi-mount, and agent-driven (LLM) use.

use crate::common::*;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS + ANTHROPIC creds + microsandbox"]
async fn e2e_sandbox_forwarder() {
    dotenvy::dotenv().ok();
    let s = stamp();
    let fname = format!("e2e-sandbox-{s}.txt");
    let content = format!("sandbox-{s}");
    let sandbox = RunEnv::sandbox(SandboxConfig {
        network: SandboxNetwork::Host,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(all_vfs())
        .build()
        .expect("build agent");
    let transcript = drive(agent, &task_for(&fname, &content)).await;
    println!(
        "--- sandbox transcript tail ---\n{}",
        tail(&transcript, 500)
    );
    assert!(
        verify(&format!("/s3/{fname}"), &content).await,
        "sandbox write not found in S3"
    );
}

/// Mirrors agent-k's lifecycle: build an agent against a persisted (by-name)
/// sandbox, use it, drop it (VM stops while idle → in-guest forwarder dies),
/// then build a *new* agent against the same sandbox — which must transparently
/// re-mount. Drive each agent once (triggers ensure_mounted), then read the
/// fixture file's byte count via `msb exec` while the agent still holds the VM up.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: re-mount across sandbox restart (needs creds + sandbox + macFUSE)"]
async fn vfs_sandbox_remount_after_restart() {
    dotenvy::dotenv().ok();
    // Fresh, unique name — avoid corrupt leftover state from prior runs.
    let name = format!("ailoy-vfs-remount-{}", stamp());
    // Set up our own S3 fixture instead of assuming a pre-existing provider file.
    let fx = Fixture::create("/s3").await;
    let path = fx.guest_path();

    async fn attach_round(name: &str, path: &str) -> i64 {
        let sandbox = RunEnv::sandbox(SandboxConfig {
            name: Some(name.into()),
            persist: true,
            network: SandboxNetwork::Host,
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
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            if let Err(e) = ev {
                eprintln!("run error: {e}");
            }
        }
        // VM is up (agent holds the handle); read via msb exec before dropping.
        msb_read_count(name, path)
        // `agent` drops here -> AgentVfs + handle drop -> VM stops.
    }

    // The bootstrap uses the static, dependency-free forwarder binary (mounts
    // /dev/fuse directly — no python/fuse3/apt), so a clean fresh sandbox mounts
    // with no setup.
    let want = fx.len() as i64;
    let n1 = attach_round(&name, &path).await;
    println!("round 1 (fresh attach): {n1} bytes");
    assert_eq!(n1, want, "round 1 mount/read should match the fixture size");

    let n2 = attach_round(&name, &path).await;
    println!("round 2 (re-attach after VM stop): {n2} bytes");
    assert_eq!(
        n2, want,
        "round 2 re-mount after restart should match the fixture size"
    );

    msb_rm(&name);
    fx.teardown().await;
}

/// Deterministic stress of the drop -> immediate-reconnect race on a persisted
/// sandbox — the agent-k lifecycle (drop runtime, recreate against the same VM)
/// with no agent/LLM variance. Each iteration acquires the VM handle, runs a
/// trivial exec, then drops the handle (which kicks off a fire-and-forget async
/// VM stop) and immediately reconnects by name. If the in-flight stop races the
/// next reconnect's force-stop/start and wedges the msb VM, the iteration blows
/// past its timeout and fails here instead of hanging the whole suite.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: stress sandbox reconnect race (needs microsandbox)"]
async fn vfs_sandbox_reconnect_race() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-race-{}", stamp());
    for i in 0..12 {
        let t0 = std::time::Instant::now();
        let iter = async {
            let env = RunEnv::sandbox(SandboxConfig {
                name: Some(name.clone()),
                persist: true,
                network: SandboxNetwork::Host,
                ..Default::default()
            })
            .await
            .expect("sandbox new");
            let handle = env.get().await.expect("get handle");
            let out = handle
                .exec_shell("echo ok".into(), Some(20))
                .await
                .expect("exec");
            assert_eq!(out.exit_code, 0, "iter {i} exec exit");
            // Drop handle (fire-and-forget async stop) then env; immediately loop
            // into the next reconnect to maximize the race window.
            drop(handle);
            drop(env);
        };
        // Allow > the bounded reconnect retry budget (start_detached_resilient:
        // 4 x 25s + backoff): a full retry recovery is success, not a wedge.
        match tokio::time::timeout(std::time::Duration::from_secs(140), iter).await {
            Ok(()) => println!("iter {i}: ok in {:.1}s", t0.elapsed().as_secs_f32()),
            Err(_) => panic!("iter {i} HUNG > 140s — reconnect race wedged the VM"),
        }
    }
    msb_rm(&name);
    println!("RACE TEST DONE (12 reconnects clean)");
}

/// Stress concurrent access through the guest mount — the scenario the worker-pool
/// forwarder exists for (a recursive grep, or parallel readers in an agent shell).
/// Launches 8 simultaneous readers of the same fixture file and asserts they
/// all return identical, complete byte counts within a bounded time. A forwarder
/// that serialized badly or deadlocked under concurrency would hang (the guest
/// `timeout` then yields short/zero counts and the assert fails) rather than wedge
/// the whole suite.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: concurrent mount access (needs creds + sandbox + macFUSE)"]
async fn vfs_concurrent_access_stress() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-concur-{}", stamp());
    let fx = Fixture::create("/s3").await;
    let path = fx.guest_path();
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: true,
        network: SandboxNetwork::Host,
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
    let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
    let mut strm = agent.run(q);
    while let Some(ev) = strm.next().await {
        if let Err(e) = ev {
            eprintln!("run error: {e}");
        }
    }
    // 8 readers in parallel; each prints its byte count. Guest-side `timeout`
    // bounds a hang so the test fails fast instead of blocking the suite.
    let script = format!("for i in $(seq 1 8); do (cat {path} | wc -c) & done; wait");
    let out = std::process::Command::new("msb")
        .args(["exec", &name, "--", "timeout", "70", "sh", "-c", &script])
        .output()
        .expect("msb exec");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let counts: Vec<i64> = stdout
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    println!("concurrent read counts: {counts:?}");
    msb_rm(&name);
    fx.teardown().await;
    assert_eq!(counts.len(), 8, "expected 8 reader results, got {counts:?}");
    let first = counts[0];
    assert_eq!(
        first,
        fx.len() as i64,
        "reads should match the fixture size: {counts:?}"
    );
    assert!(
        counts.iter().all(|&c| c == first),
        "concurrent reads returned inconsistent sizes: {counts:?}"
    );
    println!("CONCURRENT STRESS DONE ({first} bytes x8)");
}

/// The realistic agent-k scenario: an agent mounts MULTIPLE providers at once and
/// must keep accessing all of them across a VM restart. Round 1 writes a marker
/// through the s3 mount and reads real content from each *other* available
/// provider through the forwarder (notion: a page.json discovered dynamically;
/// gdrive: a readdir that round-trips). The agent drops (VM stops); round 2
/// attaches a fresh agent to the same sandbox and must read the s3 marker back
/// (write-through survived) AND re-read the other providers — proving every mount
/// re-mounts, not just one. Nothing is hardcoded: s3 uses a self-created marker,
/// notion/gdrive are discovered dynamically. (GDrive *content* reads are covered
/// by the direct-adapter smoke test; here gdrive proves readdir-over-forwarder.)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: multi-mount survives VM restart (needs s3 + notion/gdrive creds + sandbox + macFUSE)"]
async fn vfs_multimount_remount_after_restart() {
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping multimount");
        return;
    }
    // Read real content from each non-s3 provider that's configured. Each snippet
    // echoes `LABEL=<n>`; we assert n > 0 for every probed provider, in both rounds.
    let mut probes: Vec<(&str, &str)> = Vec::new();
    if has_mount("/notion") {
        probes.push((
            "NOTION",
            "p=$(ls /mnt/vfs/notion/pages 2>/dev/null | head -1); \
             [ -n \"$p\" ] && echo \"NOTION=$(cat \"/mnt/vfs/notion/pages/$p/page.json\" | wc -c)\" \
             || echo NOTION=0",
        ));
    }
    if has_mount("/gdrive") {
        probes.push((
            "GDRIVE",
            "echo \"GDRIVE=$(ls /mnt/vfs/gdrive 2>/dev/null | wc -l)\"",
        ));
    }
    if probes.is_empty() {
        eprintln!("no notion/gdrive creds — multimount needs a 2nd provider; skipping");
        return;
    }
    let probe_script = probes
        .iter()
        .map(|(_, s)| *s)
        .collect::<Vec<_>>()
        .join("; ");

    let name = format!("ailoy-vfs-multi-{}", stamp());
    let marker = format!("/mnt/vfs/s3/ailoy-multimount-{}-{}.txt", stamp(), uniq());
    let content = "multimount-ok";

    fn probe_val(stdout: &str, label: &str) -> i64 {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{label}=")))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(-1)
    }

    // Round 1: write through the s3 mount + read each other provider (VM held by agent).
    let a1 = attach_mounted_agent(&name).await;
    let r1 = std::process::Command::new("msb")
        .args([
            "exec",
            &name,
            "--",
            "sh",
            "-c",
            &format!("printf '{content}' > {marker}; {probe_script}"),
        ])
        .output()
        .expect("msb exec r1");
    let s1 = String::from_utf8_lossy(&r1.stdout);
    println!("round 1: {s1:?}, wrote s3 marker {marker}");
    for &(label, _) in &probes {
        assert!(
            probe_val(&s1, label) > 0,
            "round 1 {label} read empty/failed: {s1:?}"
        );
    }
    drop(a1); // VM stops

    // Round 2: fresh agent, same sandbox — every mount must re-mount and serve.
    let a2 = attach_mounted_agent(&name).await;
    let r2 = std::process::Command::new("msb")
        .args([
            "exec",
            &name,
            "--",
            "sh",
            "-c",
            &format!("cat {marker}; echo '|'; {probe_script}; rm -f {marker}"),
        ])
        .output()
        .expect("msb exec r2");
    let s2 = String::from_utf8_lossy(&r2.stdout);
    println!("round 2 output: {s2:?}");
    drop(a2);
    msb_rm(&name);

    assert!(
        s2.contains(content),
        "round 2: s3 write-through should survive VM restart, got {s2:?}"
    );
    for &(label, _) in &probes {
        assert!(
            probe_val(&s2, label) > 0,
            "round 2 {label} re-read empty/failed: {s2:?}"
        );
    }
    println!(
        "MULTIMOUNT DONE (s3 marker + {} survived restart)",
        probes.iter().map(|(l, _)| *l).collect::<Vec<_>>().join("+")
    );
}

/// The literal agent-k flow, end-to-end through the LLM's shell tool (not `msb
/// exec`): create an agent on a persisted sandbox and drive it once (mount comes
/// up), drop it (VM stops), then create a NEW agent on the same sandbox and have
/// the model itself read the provider file via the shell tool. Asserts the
/// recreated agent's transcript reports the fixture file's exact contents —
/// proving the agent transparently regained provider access across the VM restart
/// with no manual re-mount.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: agent-k flow — recreated agent reads provider file via shell tool across VM restart"]
async fn vfs_agent_reads_across_restart() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-agentk-{}", stamp());
    let fx = Fixture::create("/s3").await;
    let path = fx.guest_path();

    // Round 1: bring the mount up via a real agent, then drop it (VM stops).
    {
        let sandbox = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            network: SandboxNetwork::Host,
            ..Default::default()
        })
        .await
        .expect("sandbox r1");
        let agent = AgentBuilder::new(MODEL)
            .provider(provider())
            .instruction("You are a tester. Use the shell tool for everything.")
            .shell_tool()
            .runenv(sandbox)
            .vfs(all_vfs())
            .build()
            .expect("build agent r1");
        let _ = drive(agent, "Run `ls /mnt/vfs/s3` and report what you see.").await;
        // agent drops here -> VM stops
    }

    // Round 2: a *new* agent on the same persisted sandbox; the model reads the
    // provider file through the shell tool after the restart.
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: true,
        network: SandboxNetwork::Host,
        ..Default::default()
    })
    .await
    .expect("sandbox r2");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(all_vfs())
        .build()
        .expect("build agent r2");
    let task = format!(
        "Run exactly this one shell command and then report its exact output \
         verbatim, with no extra commentary: cat {path}"
    );
    let transcript = drive(agent, &task).await;
    println!(
        "--- agent-k round 2 transcript tail ---\n{}",
        tail(&transcript, 700)
    );
    msb_rm(&name);
    fx.teardown().await;
    assert!(
        transcript.contains(&fx.content),
        "recreated agent should read the provider file via its shell tool across \
         the VM restart (expected fixture content {:?} in transcript): {transcript:?}",
        fx.content
    );
    println!("AGENT-K FLOW OK: recreated agent read provider file across VM restart");
}

/// Multi-agent agent-k stress: N agents, each on its own persisted sandbox, all
/// going through attach → mount → drop → reconnect → read CONCURRENTLY. This is
/// the worst case for microsandbox's shared SQLite state layer (the contention
/// that intermittently hung create/reconnect), so it directly validates that the
/// bounded+retry sandbox-acquire path keeps provider access working under the
/// concurrency agent-k actually produces. Every agent must read the page in both
/// rounds; a wedged lifecycle would surface as a zero/by the bounded error.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: N concurrent agents re-mount across VM restart (creds + sandbox + macFUSE)"]
async fn vfs_concurrent_agents_remount() {
    dotenvy::dotenv().ok();
    const N: usize = 4;
    // One shared S3 fixture; all agents read it (read-only, so sharing is fine).
    let fx = Fixture::create("/s3").await;

    async fn one_agent(idx: usize, stamp: u64, path: String) -> (i64, i64) {
        let name = format!("ailoy-vfs-cc-{stamp}-{idx}");
        async fn attach_round(name: &str, path: &str) -> i64 {
            let sandbox = RunEnv::sandbox(SandboxConfig {
                name: Some(name.into()),
                persist: true,
                network: SandboxNetwork::Host,
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
            let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
            let mut strm = agent.run(q);
            while let Some(ev) = strm.next().await {
                if let Err(e) = ev {
                    eprintln!("[{name}] run error: {e}");
                }
            }
            let out = std::process::Command::new("msb")
                .args([
                    "exec",
                    name,
                    "--",
                    "sh",
                    "-c",
                    &format!("cat {path} | wc -c"),
                ])
                .output()
                .expect("msb exec");
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(0)
        }
        let n1 = attach_round(&name, path.as_str()).await;
        let n2 = attach_round(&name, path.as_str()).await;
        msb_rm(&name);
        (n1, n2)
    }

    let stamp = stamp();
    let mut handles = Vec::new();
    for i in 0..N {
        let path = fx.guest_path();
        handles.push(tokio::spawn(one_agent(i, stamp, path)));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.expect("agent task panicked"));
    }
    println!("concurrent results (n1,n2) per agent: {results:?}");
    let want = fx.len() as i64;
    fx.teardown().await;
    for (i, (n1, n2)) in results.iter().enumerate() {
        assert_eq!(
            *n1, want,
            "agent {i} round 1 (fresh attach) should match fixture size"
        );
        assert_eq!(
            *n2, want,
            "agent {i} round 2 (re-mount after restart) should match fixture size"
        );
    }
    println!("CONCURRENT AGENTS OK ({N} agents, both rounds across restart)");
}

/// getattr/stat size correctness through the forwarder for a *rendered* file
/// whose directory listing reports size 0 (a Notion page.json). That size-0
/// listing once made reads clamp to 0; stat now verifies the real size. Asserts
/// the size reported by `stat`/`ls -l` (a getattr round-trip through the
/// forwarder) equals the actual content length read back, both > 0. Needs a
/// notion mount with at least one shared page.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: stat size matches content for a rendered file through the forwarder (creds + sandbox + macFUSE)"]
async fn vfs_stat_size_through_mount() {
    dotenvy::dotenv().ok();
    if !has_mount("/notion") {
        eprintln!("no notion creds — skipping (needs a rendered/size-0-listing file)");
        return;
    }
    let name = format!("ailoy-vfs-stat-{}", stamp());
    let agent = attach_mounted_agent(&name).await;
    let out = msb_exec(
        &name,
        "p=$(ls /mnt/vfs/notion/pages 2>/dev/null | head -1); \
         [ -z \"$p\" ] && { echo NO_PAGE; exit 0; }; \
         f=\"/mnt/vfs/notion/pages/$p/page.json\"; \
         echo \"STAT=$(stat -c %s \"$f\" 2>/dev/null || ls -ln \"$f\" | awk '{print $5}')\"; \
         echo \"READ=$(cat \"$f\" | wc -c)\"",
    );
    println!("stat-size probe: {out:?}");
    drop(agent);
    msb_rm(&name);
    if out.contains("NO_PAGE") {
        eprintln!("no notion page shared with the integration — skipping");
        return;
    }
    let parse = |k: &str| -> i64 {
        out.lines()
            .find_map(|l| l.strip_prefix(k))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(-1)
    };
    let stat = parse("STAT=");
    let read = parse("READ=");
    assert!(
        read > 0,
        "read of the rendered page.json should be non-empty: {out:?}"
    );
    assert_eq!(
        stat, read,
        "stat/getattr size ({stat}) must match the actual content length ({read}) \
         — a size-0 listing must not clamp the reported size"
    );
    println!("STAT-SIZE OK (getattr {stat} == read {read})");
}

/// Domain `.cmd` write routed through the in-guest forwarder: write a
/// page-create body to `/mnt/vfs/notion/.cmd/page-create`. The forward server
/// strips `/.cmd/<op>` and dispatches to `Resource::command`, so this exercises
/// the write→command seam end to end (not the direct adapter, which the
/// providers.rs smoke test covers). The deterministic guarantee asserted here is
/// liveness: the `.cmd` write reaches the host command handler and returns
/// without wedging the mount (the bug class is an indefinite hang). If the
/// integration has "Insert content" capability a page is actually created (a
/// logged bonus); ACL rejection is fine. Needs a notion mount + a shared page.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: domain .cmd write routed through the forwarder (creds + sandbox + macFUSE)"]
async fn vfs_cmd_write_through_forwarder() {
    dotenvy::dotenv().ok();
    if !has_mount("/notion") {
        eprintln!("no notion creds — skipping");
        return;
    }
    let name = format!("ailoy-vfs-cmd-{}", stamp());
    let agent = attach_mounted_agent(&name).await;
    // Discover a parent page in-guest, build the page-create body with its id, and
    // write it to the `.cmd` control path. Then confirm the mount is still alive —
    // proving the write routed to command() without wedging the forwarder.
    let out = msb_exec(
        &name,
        "p=$(ls /mnt/vfs/notion/pages 2>/dev/null | head -1); \
         [ -z \"$p\" ] && { echo NO_PAGE; exit 0; }; \
         id=${p##*__}; \
         json='{\"parent\":{\"page_id\":\"'\"$id\"'\"},\"properties\":{\"title\":[{\"text\":{\"content\":\"ailoy vfs cmd-through-fwd\"}}]}}'; \
         printf '%s' \"$json\" > /mnt/vfs/notion/.cmd/page-create 2>/tmp/cmderr; \
         echo \"WRITE_RC=$?\"; \
         echo \"RESULT=$(cat /mnt/vfs/notion/.cmd/page-create 2>/dev/null | head -c 400)\"; \
         (ls /mnt/vfs/notion/pages >/dev/null 2>&1 && echo MOUNT_ALIVE || echo MOUNT_DEAD); \
         echo \"ERR=$(cat /tmp/cmderr 2>/dev/null)\"",
    );
    println!("cmd-through-forwarder: {out:?}");
    drop(agent);
    msb_rm(&name);
    if out.contains("NO_PAGE") {
        eprintln!("no notion page shared with the integration — skipping");
        return;
    }
    // Liveness: the `.cmd` write routed to the host command handler and the mount
    // survived (didn't hang/wedge). A capability/ACL rejection is acceptable.
    assert!(
        out.contains("MOUNT_ALIVE"),
        "mount wedged or unreachable after a .cmd write — routing/liveness broken: {out:?}"
    );
    if out.contains("WRITE_RC=0") {
        // C4: the command's JSON result is readable back from the `.cmd` path
        // (here: the created page's id), enabling create→block-append.
        assert!(
            out.contains("RESULT=") && out.contains("\"id\""),
            "C4: page-create result was not readable back from the .cmd path: {out:?}"
        );
        println!("CMD-THROUGH-FORWARDER OK (page-create routed + result read back)");
    } else {
        eprintln!(
            "NOTE: .cmd write routed through the forwarder but the op was rejected \
             (integration capability/ACL); mount stayed live — soft pass"
        );
    }
}

/// Holds a NAMED sandbox (`ailoy-vfs-inspect`) with every configured provider
/// mounted under /mnt/vfs, so you can poke it from another terminal:
///   msb exec ailoy-vfs-inspect -- sh -c 'ls /mnt/vfs/s3'
///   msb exec ailoy-vfs-inspect -- sh -c 'ls /mnt/vfs/notion/pages'
///   msb exec ailoy-vfs-inspect -- sh -c 'ls /mnt/vfs/gdrive | head'
///
/// Starts clean before booting: a VM from a prior run can outlive its process
/// (the sandbox supervisor survives a non-graceful exit), and the fixed name
/// would otherwise reattach to that STALE VM — whose original host forward
/// server is gone — so the mount serves nothing. Tearing the stale instance down
/// first guarantees a fresh boot, which is the common cause of "it blocks" here.
///
/// Note: the microsandbox guest→host channel can still degrade intermittently on
/// a long-lived VM (host-side, not ours — the host forward server stays healthy).
/// With the forwarder's bounded resolve+connect, a degraded channel now makes
/// `msb exec` into the mount return an error within seconds rather than hanging;
/// re-run this test to get a fresh VM if that happens.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "interactive: holds a named sandbox up for manual msb exec inspection"]
async fn vfs_inspect_sandbox() {
    dotenvy::dotenv().ok();
    let name = "ailoy-vfs-inspect";

    // Clear any stale instance from a prior run so we always boot fresh (see above).
    msb_rm(name);

    let agent = attach_mounted_agent(name).await;

    println!("\n================ VFS INSPECT READY ================");
    println!("sandbox name : {name}");
    println!("guest mounts : /mnt/vfs/{{s3,notion,gdrive}} (configured providers)");
    println!("inspect from another terminal:");
    println!("  msb exec {name} -- sh -c 'ls -la /mnt/vfs'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/s3'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/notion/pages'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/gdrive | head'");
    println!("Ctrl-C to tear down (cleans up the sandbox automatically).");
    println!("===================================================\n");

    // Hold the VM + host forward server up until Ctrl-C, then tear the sandbox
    // down so it doesn't leak — a SIGINT-killed process would otherwise orphan
    // the VM (its supervisor outlives the process). `tokio::signal::ctrl_c`
    // intercepts the first Ctrl-C; cleanup runs, then the test returns.
    let _ = tokio::signal::ctrl_c().await;
    println!("\n[inspect] Ctrl-C — tearing down {name}…");
    drop(agent); // AgentVfs Drop stops the VM + host forward server
    msb_rm(name); // ensure it's stopped + removed (no orphan)
    println!("[inspect] done.");
}
