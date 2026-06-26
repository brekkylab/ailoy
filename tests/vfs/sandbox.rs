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
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(vfs_config())
        .build()
        .expect("build agent");
    let transcript = drive(agent, &task_for(&fname, &content)).await;
    println!(
        "--- sandbox transcript tail ---\n{}",
        tail(&transcript, 500)
    );
    assert!(
        verify_s3(&fname, &content).await,
        "sandbox write not found in S3"
    );
}

/// Mirrors agent-k's lifecycle: build an agent against a persisted (by-name)
/// sandbox, use it, drop it (VM stops while idle → in-guest forwarder dies),
/// then build a *new* agent against the same sandbox — which must transparently
/// re-mount. Drive each agent once (triggers ensure_mounted), then read the
/// page.json byte count via `msb exec` while the agent still holds the VM up.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: re-mount across sandbox restart (needs creds + sandbox + macFUSE)"]
async fn vfs_sandbox_remount_after_restart() {
    dotenvy::dotenv().ok();
    // Fresh, unique name — avoid corrupt leftover state from prior runs.
    let name = format!("ailoy-vfs-remount-{}", stamp());
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";

    async fn attach_round(name: &str, page: &str) -> i64 {
        let sandbox = RunEnv::sandbox(SandboxConfig {
            name: Some(name.into()),
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
            .vfs(notion_vfs())
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
        let out = std::process::Command::new("msb")
            .args([
                "exec",
                name,
                "--",
                "sh",
                "-c",
                &format!("cat {page} | wc -c"),
            ])
            .output()
            .expect("msb exec");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or(0)
        // `agent` drops here -> AgentVfs + handle drop -> VM stops.
    }

    // The bootstrap uses the static, dependency-free forwarder binary (mounts
    // /dev/fuse directly — no python/fuse3/apt), so a clean fresh sandbox mounts
    // with no setup.
    let n1 = attach_round(&name, page).await;
    println!("round 1 (fresh attach): {n1} bytes");
    assert!(n1 > 0, "round 1 mount/read should be non-empty");

    let n2 = attach_round(&name, page).await;
    println!("round 2 (re-attach after VM stop): {n2} bytes");
    assert!(n2 > 0, "round 2 re-mount after restart should be non-empty");

    let _ = std::process::Command::new("msb")
        .args(["stop", &name])
        .status();
    let _ = std::process::Command::new("msb")
        .args(["rm", &name])
        .status();
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
                allow_host_egress: true,
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
    let _ = std::process::Command::new("msb")
        .args(["stop", &name])
        .status();
    let _ = std::process::Command::new("msb")
        .args(["rm", &name])
        .status();
    println!("RACE TEST DONE (12 reconnects clean)");
}

/// Stress concurrent access through the guest mount — the scenario the worker-pool
/// forwarder exists for (a recursive grep, or parallel readers in an agent shell).
/// Launches 8 simultaneous readers of the same Notion page.json and asserts they
/// all return identical, complete byte counts within a bounded time. A forwarder
/// that serialized badly or deadlocked under concurrency would hang (the guest
/// `timeout` then yields short/zero counts and the assert fails) rather than wedge
/// the whole suite.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: concurrent mount access (needs creds + sandbox + macFUSE)"]
async fn vfs_concurrent_access_stress() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-concur-{}", stamp());
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";
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
        .vfs(notion_vfs())
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
    let script = format!("for i in $(seq 1 8); do (cat {page} | wc -c) & done; wait");
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
    let _ = std::process::Command::new("msb")
        .args(["stop", &name])
        .status();
    let _ = std::process::Command::new("msb")
        .args(["rm", &name])
        .status();
    assert_eq!(counts.len(), 8, "expected 8 reader results, got {counts:?}");
    let first = counts[0];
    assert!(first > 0, "reads returned empty: {counts:?}");
    assert!(
        counts.iter().all(|&c| c == first),
        "concurrent reads returned inconsistent sizes: {counts:?}"
    );
    println!("CONCURRENT STRESS DONE ({first} bytes x8)");
}

/// The realistic agent-k scenario: an agent mounts MULTIPLE providers at once
/// (notion + s3) and must keep accessing all of them across a VM restart. Round 1
/// writes a marker through the s3 mount and reads a notion page; the agent drops
/// (VM stops); round 2 attaches a fresh agent to the same persisted sandbox and
/// must read the s3 marker back (write-through to the provider survived the
/// restart) AND re-read notion — proving every mount re-mounts, not just one.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: multi-mount (notion+s3) survives VM restart (needs creds + sandbox + macFUSE)"]
async fn vfs_multimount_remount_after_restart() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-multi-{}", stamp());
    let marker = format!("/mnt/vfs/s3/ailoy-multimount-{}.txt", stamp());
    let content = "multimount-ok";
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";

    fn cfg() -> VfsConfig {
        VfsConfig {
            mounts: vec![
                MountSpec {
                    prefix: "/notion".into(),
                    provider: ProviderConfig::Notion(ailoy::vfs::NotionConfig {
                        api_key: std::env::var("NOTION_API_KEY").unwrap(),
                    }),
                },
                MountSpec {
                    prefix: "/s3".into(),
                    provider: ProviderConfig::S3(s3_config()),
                },
            ],
        }
    }

    async fn attach(name: &str) -> Agent {
        let sandbox = RunEnv::sandbox(SandboxConfig {
            name: Some(name.into()),
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
            .vfs(cfg())
            .build()
            .expect("build agent");
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            if let Err(e) = ev {
                eprintln!("run error: {e}");
            }
        }
        drop(strm); // release the borrow on `agent` so it can be returned
        agent
    }

    // Round 1: write through the s3 mount + read the notion mount (VM held by agent).
    let a1 = attach(&name).await;
    let r1 = std::process::Command::new("msb")
        .args([
            "exec",
            &name,
            "--",
            "sh",
            "-c",
            &format!("printf '{content}' > {marker} && cat {page} | wc -c"),
        ])
        .output()
        .expect("msb exec r1");
    let n1: i64 = String::from_utf8_lossy(&r1.stdout)
        .split_whitespace()
        .last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    println!("round 1: notion={n1} bytes, wrote s3 marker {marker}");
    assert!(n1 > 0, "round 1 notion read should be non-empty");
    drop(a1); // VM stops

    // Round 2: fresh agent, same sandbox — both mounts must re-mount and serve.
    let a2 = attach(&name).await;
    let r2 = std::process::Command::new("msb")
        .args([
            "exec",
            &name,
            "--",
            "sh",
            "-c",
            &format!("cat {marker}; echo '|'; cat {page} | wc -c; rm -f {marker}"),
        ])
        .output()
        .expect("msb exec r2");
    let stdout = String::from_utf8_lossy(&r2.stdout);
    println!("round 2 output: {stdout:?}");
    let n2: i64 = stdout
        .split_whitespace()
        .last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    drop(a2);
    let _ = std::process::Command::new("msb")
        .args(["stop", &name])
        .status();
    let _ = std::process::Command::new("msb")
        .args(["rm", &name])
        .status();

    assert!(
        stdout.contains(content),
        "round 2: s3 write-through should survive VM restart, got {stdout:?}"
    );
    assert!(n2 > 0, "round 2 notion re-read should be non-empty");
    println!("MULTIMOUNT DONE (s3 marker + notion both survived restart)");
}

/// The literal agent-k flow, end-to-end through the LLM's shell tool (not `msb
/// exec`): create an agent on a persisted sandbox and drive it once (mount comes
/// up), drop it (VM stops), then create a NEW agent on the same sandbox and have
/// the model itself read the provider file via the shell tool. Asserts the
/// recreated agent's transcript reports the file's byte count — proving the agent
/// transparently regained provider access across the VM restart with no manual
/// re-mount.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: agent-k flow — recreated agent reads provider file via shell tool across VM restart"]
async fn vfs_agent_reads_across_restart() {
    dotenvy::dotenv().ok();
    let name = format!("ailoy-vfs-agentk-{}", stamp());
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";

    // Round 1: bring the mount up via a real agent, then drop it (VM stops).
    {
        let sandbox = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            allow_host_egress: true,
            ..Default::default()
        })
        .await
        .expect("sandbox r1");
        let agent = AgentBuilder::new(MODEL)
            .provider(provider())
            .instruction("You are a tester. Use the shell tool for everything.")
            .shell_tool()
            .runenv(sandbox)
            .vfs(notion_vfs())
            .build()
            .expect("build agent r1");
        let _ = drive(agent, "Run `ls /mnt/vfs/notion` and report what you see.").await;
        // agent drops here -> VM stops
    }

    // Round 2: a *new* agent on the same persisted sandbox; the model reads the
    // provider file through the shell tool after the restart.
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.clone()),
        persist: true,
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox r2");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(notion_vfs())
        .build()
        .expect("build agent r2");
    let task = format!(
        "Run exactly this one shell command and then tell me, as a plain number, \
         the single integer it prints: cat {page} | wc -c"
    );
    let transcript = drive(agent, &task).await;
    println!(
        "--- agent-k round 2 transcript tail ---\n{}",
        tail(&transcript, 700)
    );
    let _ = std::process::Command::new("msb")
        .args(["stop", &name])
        .status();
    let _ = std::process::Command::new("msb")
        .args(["rm", &name])
        .status();
    assert!(
        transcript.contains("489"),
        "recreated agent should read the provider file via its shell tool across \
         the VM restart (expected byte count 489 in transcript): {transcript:?}"
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
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";

    async fn one_agent(idx: usize, stamp: u64, page: String) -> (i64, i64) {
        let name = format!("ailoy-vfs-cc-{stamp}-{idx}");
        async fn attach_round(name: &str, page: &str) -> i64 {
            let sandbox = RunEnv::sandbox(SandboxConfig {
                name: Some(name.into()),
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
                .vfs(notion_vfs())
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
                    &format!("cat {page} | wc -c"),
                ])
                .output()
                .expect("msb exec");
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(0)
        }
        let n1 = attach_round(&name, page.as_str()).await;
        let n2 = attach_round(&name, page.as_str()).await;
        let _ = std::process::Command::new("msb")
            .args(["stop", &name])
            .status();
        let _ = std::process::Command::new("msb")
            .args(["rm", &name])
            .status();
        (n1, n2)
    }

    let stamp = stamp();
    let mut handles = Vec::new();
    for i in 0..N {
        let page = page.to_string();
        handles.push(tokio::spawn(one_agent(i, stamp, page)));
    }
    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.expect("agent task panicked"));
    }
    println!("concurrent results (n1,n2) per agent: {results:?}");
    for (i, (n1, n2)) in results.iter().enumerate() {
        assert!(*n1 > 0, "agent {i} round 1 (fresh attach) returned empty");
        assert!(
            *n2 > 0,
            "agent {i} round 2 (re-mount after restart) returned empty"
        );
    }
    println!("CONCURRENT AGENTS OK ({N} agents, both rounds across restart)");
}

/// Brings up a NAMED sandbox with S3 + Notion + GDrive mounted under /mnt/vfs,
/// triggers the in-guest forwarder mount, then sleeps so you can inspect it:
///   msb ls
///   msb exec ailoy-vfs-inspect -- sh -c 'ls /mnt/vfs; ls /mnt/vfs/s3; ls /mnt/vfs/notion/pages; ls /mnt/vfs/gdrive'
#[tokio::test(flavor = "multi_thread")]
#[ignore = "interactive: holds a named sandbox up for manual msb exec inspection"]
async fn vfs_inspect_sandbox() {
    dotenvy::dotenv().ok();

    let name = "ailoy-vfs-inspect";
    let sandbox = RunEnv::sandbox(SandboxConfig {
        name: Some(name.into()),
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox create");

    let mut agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool.")
        .shell_tool()
        .runenv(sandbox)
        .vfs(all_vfs())
        .build()
        .expect("build agent");

    // First run triggers the in-guest mount (ensure_vfs_mounted). Keep `agent`
    // alive afterwards so the strong runenv handle keeps the VM (and mount) up.
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

    println!("\n================ VFS INSPECT READY ================");
    println!("sandbox name : {name}");
    println!("guest mounts : /mnt/vfs/{{s3,notion,gdrive}}");
    println!("inspect from another terminal:");
    println!("  msb ls");
    println!("  msb exec {name} -- sh -c 'mount | grep fuse; ls -la /mnt/vfs'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/s3'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/notion/pages'");
    println!("  msb exec {name} -- sh -c 'ls /mnt/vfs/gdrive | head'");
    println!("sleeping 1h — Ctrl-C this process to tear down.");
    println!("===================================================\n");

    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
}
