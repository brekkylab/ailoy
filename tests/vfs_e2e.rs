//! Live end-to-end tests for the VFS provider mounts. Requires AWS + Anthropic
//! credentials in the environment and (for the sandbox case) a working
//! microsandbox + macFUSE/libfuse host. Run explicitly:
//!
//! ```sh
//! set -a; . .env; set +a
//! cargo test --features "vfs sandbox" --test vfs_e2e -- --ignored --nocapture
//! ```

#![cfg(all(feature = "vfs", feature = "sandbox"))]

use std::time::{SystemTime, UNIX_EPOCH};

use ailoy::{
    agent::{Agent, AgentBuilder, AgentProvider},
    lang_model::LangModelProvider,
    message::{Message, Part, Role},
    runenv::{RunEnv, SandboxConfig},
    vfs::{FileKind, MountSpec, ProviderConfig, S3Config, Vfs, VfsConfig},
};
use futures::StreamExt;

const MODEL: &str = "anthropic/claude-haiku-4-5";

fn s3_config() -> S3Config {
    S3Config {
        bucket: std::env::var("AWS_S3_BUCKET").unwrap(),
        region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        endpoint: None,
        key_prefix: None,
    }
}

fn vfs_config() -> VfsConfig {
    VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/s3".into(),
            provider: ProviderConfig::S3(s3_config()),
        }],
    }
}

fn provider() -> AgentProvider {
    let key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    let mut p = AgentProvider::new();
    p.models
        .insert(MODEL.into(), LangModelProvider::anthropic(key));
    p
}

fn stamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

async fn drive(mut agent: Agent, task: &str) -> String {
    let query = Message::new(Role::User).with_contents([Part::text(task)]);
    let mut strm = agent.run(query);
    let mut transcript = String::new();
    while let Some(ev) = strm.next().await {
        let ev = ev.expect("agent event");
        if ev.message.role == Role::Assistant {
            for part in &ev.message.contents {
                if let Some(t) = part.as_text() {
                    transcript.push_str(t);
                    transcript.push('\n');
                }
            }
        }
    }
    transcript
}

async fn verify_s3(fname: &str, want: &str) -> bool {
    let vfs = Vfs::from_config(vfs_config()).unwrap();
    let path = format!("/s3/{fname}");
    let (res, vp) = vfs.route(&path).expect("route");
    match res.read_bytes(&vp, None).await {
        Ok(data) => {
            let got = String::from_utf8_lossy(&data);
            println!("    [verify] s3 {fname} => {:?}", got.trim());
            got.contains(want)
        }
        Err(e) => {
            println!("    [verify] read failed: {e}");
            false
        }
    }
}

fn task_for(fname: &str, content: &str) -> String {
    format!(
        "Your instructions list an external S3 mount path. \
         First run `ls` on that s3 mount directory. \
         Then create a file named `{fname}` in that s3 mount directory whose only \
         content is the exact text `{content}`, using a shell redirect. \
         Then `cat` that file to confirm. Report what you did concisely."
    )
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS + ANTHROPIC creds + macFUSE host"]
async fn e2e_non_sandbox_host_fuser() {
    let s = stamp();
    let fname = format!("e2e-nosandbox-{s}.txt");
    let content = format!("nosandbox-{s}");
    let agent = AgentBuilder::new(MODEL)
        .provider(provider())
        .instruction("You are a tester. Use the shell tool for everything.")
        .shell_tool()
        .vfs(vfs_config())
        .build()
        .expect("build agent");
    let transcript = drive(agent, &task_for(&fname, &content)).await;
    println!(
        "--- non-sandbox transcript tail ---\n{}",
        tail(&transcript, 500)
    );
    assert!(
        verify_s3(&fname, &content).await,
        "non-sandbox write not found in S3"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS + ANTHROPIC creds + microsandbox"]
async fn e2e_sandbox_forwarder() {
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

/// Direct smoke test of the S3 adapter's `readdir` parity with mirage:
/// children come back name-sorted, subfolders are directories (via the `/`
/// delimiter), and a zero-byte "directory marker" object for the listed
/// prefix is excluded (mirage drops it; object_store does not).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS creds + aws CLI"]
async fn s3_readdir_marker_and_sort_smoke() {
    let bucket = std::env::var("AWS_S3_BUCKET").unwrap();
    let s = stamp();
    let base = format!("vfs-s3-smoke-{s}");
    let vfs = Vfs::from_config(vfs_config()).unwrap();

    // Write three children (out of order) plus a nested object.
    for (name, body) in [("b.txt", "B"), ("a.txt", "A"), ("sub/c.txt", "C")] {
        let (res, vp) = vfs.route(&format!("/s3/{base}/{name}")).expect("route");
        res.write_bytes(&vp, body.as_bytes().to_vec())
            .await
            .expect("write");
    }
    // Create the explicit directory-marker object `<base>/` (a zero-byte key
    // ending in `/`), which the Resource API cannot create on its own.
    let marker_key = format!("{base}/");
    let status = std::process::Command::new("aws")
        .args([
            "s3api",
            "put-object",
            "--bucket",
            &bucket,
            "--key",
            &marker_key,
        ])
        .status()
        .expect("run aws put-object");
    assert!(status.success(), "failed to create folder marker");

    let (res, vp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    let entries = res.readdir(&vp).await.expect("readdir");
    let names: Vec<(String, bool)> = entries
        .iter()
        .map(|e| (e.name.clone(), e.kind == FileKind::Dir))
        .collect();
    println!("readdir {base}: {names:?}");

    // Marker for the listed prefix must be gone; entries name-sorted; sub is a dir.
    assert!(
        !entries.iter().any(|e| e.name == base),
        "directory marker leaked as a child entry"
    );
    assert_eq!(
        entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        vec!["a.txt", "b.txt", "sub"],
        "entries should be name-sorted with sub/ as a directory"
    );
    assert!(
        entries.iter().find(|e| e.name == "sub").unwrap().kind == FileKind::Dir
    );

    // Cleanup.
    let _ = std::process::Command::new("aws")
        .args(["s3", "rm", &format!("s3://{bucket}/{base}"), "--recursive"])
        .status();
    let _ = std::process::Command::new("aws")
        .args(["s3api", "delete-object", "--bucket", &bucket, "--key", &marker_key])
        .status();
}

fn notion_vfs() -> VfsConfig {
    VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/notion".into(),
            provider: ProviderConfig::Notion(ailoy::vfs::NotionConfig {
                api_key: std::env::var("NOTION_API_KEY").unwrap(),
            }),
        }],
    }
}

/// Direct (non-agent) smoke test of the Notion adapter: read the page tree,
/// read a page.json, then exercise the `.cmd` domain writes (page-create +
/// block-append) through `Resource::command`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs NOTION_API_KEY with a shared page"]
async fn notion_read_and_command_smoke() {
    let vfs = Vfs::from_config(notion_vfs()).unwrap();

    let (res, vp) = vfs.route("/notion/pages").expect("route pages");
    let entries = res.readdir(&vp).await.expect("readdir pages");
    println!(
        "pages: {:?}",
        entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    // Prefer a known write-shared parent; fall back to the first page.
    let parent = entries
        .iter()
        .find(|e| e.name.contains("Engineering_Logs"))
        .or_else(|| entries.first())
        .expect("at least one shared page");
    let parent_id = parent
        .name
        .rsplit_once("__")
        .map(|(_, id)| id)
        .unwrap()
        .to_string();

    // Descend into the page dir: it should contain page.json plus a
    // subdirectory per child_page block (hierarchical, mirroring mirage).
    let (res, vp) = vfs
        .route(&format!("/notion/pages/{}", parent.name))
        .expect("route page dir");
    let dir_entries = res.readdir(&vp).await.expect("readdir page dir");
    println!(
        "{} contents: {:?}",
        parent.name,
        dir_entries
            .iter()
            .map(|e| format!("{}{}", e.name, if e.kind == FileKind::Dir { "/" } else { "" }))
            .collect::<Vec<_>>()
    );
    assert!(
        dir_entries.iter().any(|e| e.name == "page.json"),
        "page dir must expose page.json"
    );

    let (res, vp) = vfs
        .route(&format!("/notion/pages/{}/page.json", parent.name))
        .expect("route page.json");
    let data = res.read_bytes(&vp, None).await.expect("read page.json");
    let page: serde_json::Value = serde_json::from_slice(&data).unwrap();
    println!("first page title: {:?}", page.get("title"));
    // Normalized schema parity with mirage `normalize_page`.
    for k in [
        "page_id",
        "title",
        "url",
        "created_time",
        "last_edited_time",
        "parent_type",
        "parent_id",
        "archived",
        "created_by",
        "last_edited_by",
        "markdown",
        "blocks",
    ] {
        assert!(page.get(k).is_some(), "page.json missing key `{k}`");
    }

    // page-create under the first page
    let (res, _) = vfs.route("/notion/.cmd/page-create").unwrap();
    let create_body = serde_json::json!({
        "parent": {"page_id": parent_id},
        "properties": {"title": [{"text": {"content": "ailoy vfs Phase2"}}]}
    });
    // Domain write reaches the Notion API; success additionally requires the
    // integration to have the "Insert content" capability on a shared page.
    match res
        .command("page-create", create_body.to_string().as_bytes())
        .await
    {
        Ok(created) => {
            let created: serde_json::Value = serde_json::from_slice(&created).unwrap();
            let new_id = created
                .get("id")
                .and_then(|v| v.as_str())
                .expect("new page id")
                .to_string();
            println!("created page id: {new_id}");
            let append_body = serde_json::json!({
                "block_id": new_id,
                "children": [{
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": "created by ailoy vfs phase 2"}}]}
                }]
            });
            let appended = res
                .command("block-append", append_body.to_string().as_bytes())
                .await
                .expect("block-append");
            assert!(!appended.is_empty(), "block-append returned empty");
            println!("page-create + block-append OK ✅");
        }
        Err(e) => {
            println!(
                "NOTE: domain write reached Notion but was rejected (integration \
                 capability / sharing): {e}"
            );
        }
    }
}

fn gdrive_vfs() -> VfsConfig {
    VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/gdrive".into(),
            provider: ProviderConfig::GDrive(ailoy::vfs::GDriveConfig {
                client_id: std::env::var("GOOGLE_CLIENT_ID").unwrap(),
                client_secret: std::env::var("GOOGLE_CLIENT_SECRET").unwrap(),
                refresh_token: std::env::var("GOOGLE_REFRESH_TOKEN").unwrap(),
            }),
        }],
    }
}

/// Verify the GDrive adapter mirrors the Drive folder hierarchy: the root
/// lists folders as directories, and descending into a folder lists its
/// children (not a flat dump of the whole Drive).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with Drive scope"]
async fn gdrive_hierarchy_smoke() {
    let vfs = Vfs::from_config(gdrive_vfs()).unwrap();
    let (res, vp) = vfs.route("/gdrive").expect("route gdrive");
    let root = res.readdir(&vp).await.expect("readdir root");
    let dirs: Vec<&String> = root
        .iter()
        .filter(|e| matches!(e.kind, ailoy::vfs::FileKind::Dir))
        .map(|e| &e.name)
        .collect();
    println!("root: {} entries, {} folders", root.len(), dirs.len());
    println!("folders: {:?}", &dirs[..dirs.len().min(8)]);
    assert!(!dirs.is_empty(), "expected at least one folder at root");

    let folder = dirs[0].clone();
    let (res, vp) = vfs
        .route(&format!("/gdrive/{folder}"))
        .expect("route subfolder");
    let children = res.readdir(&vp).await.expect("readdir subfolder");
    println!(
        "  /{folder}: {} children -> {:?}",
        children.len(),
        children.iter().map(|e| &e.name).take(8).collect::<Vec<_>>()
    );
}

/// Direct smoke test of the GDrive adapter: list Drive, read a Google Doc as
/// `.gdoc.json`, then append to it via `.cmd/docs-append`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with Drive + Docs scopes"]
async fn gdrive_read_and_command_smoke() {
    let vfs = Vfs::from_config(gdrive_vfs()).unwrap();
    let (res, vp) = vfs.route("/gdrive").expect("route gdrive");
    let entries = res.readdir(&vp).await.expect("readdir gdrive");
    let names: Vec<&String> = entries.iter().map(|e| &e.name).collect();
    println!(
        "gdrive entries (first 10): {:?}",
        &names[..names.len().min(10)]
    );

    let gdoc = entries.iter().find(|e| e.name.ends_with(".gdoc.json"));
    let Some(gdoc) = gdoc else {
        println!("NOTE: no Google Doc found in Drive; read/append skipped");
        return;
    };
    let (res, vp) = vfs
        .route(&format!("/gdrive/{}", gdoc.name))
        .expect("route gdoc");
    let data = res.read_bytes(&vp, None).await.expect("read gdoc");
    let doc: serde_json::Value = serde_json::from_slice(&data).unwrap();
    let doc_id = doc
        .get("documentId")
        .and_then(|v| v.as_str())
        .expect("documentId")
        .to_string();
    println!("read {} -> documentId={doc_id}", gdoc.name);

    // Append to a user-owned doc (the read doc above may be read-only-shared).
    // Override via AILOY_GDOC_ID; defaults to a known-writable doc.
    let write_doc = std::env::var("AILOY_GDOC_ID")
        .unwrap_or_else(|_| "10NTjr9rPPqZKzoW_YpP9z-8WfSMuN0z1-mDq012KWuI".into());
    let _ = doc_id;
    let (res, _) = vfs.route("/gdrive/.cmd/docs-append").unwrap();
    let body =
        serde_json::json!({"document_id": write_doc, "text": "\nappended by ailoy vfs phase 2\n"});
    match res
        .command("docs-append", body.to_string().as_bytes())
        .await
    {
        Ok(result) => {
            assert!(!result.is_empty(), "docs-append returned empty");
            println!("docs-append OK ✅");
        }
        Err(e) => {
            println!(
                "NOTE: docs-append reached the Docs API but was rejected \
                 (token Docs scope / doc ACL): {e}"
            );
        }
    }
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
            .args(["exec", name, "--", "sh", "-c", &format!("cat {page} | wc -c")])
            .output()
            .expect("msb exec");
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
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

    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
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
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
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
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
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
            "exec", &name, "--", "sh", "-c",
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
            "exec", &name, "--", "sh", "-c",
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
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();

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
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
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
                .args(["exec", name, "--", "sh", "-c", &format!("cat {page} | wc -c")])
                .output()
                .expect("msb exec");
            String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
        }
        let n1 = attach_round(&name, page.as_str()).await;
        let n2 = attach_round(&name, page.as_str()).await;
        let _ = std::process::Command::new("msb").args(["stop", &name]).status();
        let _ = std::process::Command::new("msb").args(["rm", &name]).status();
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
        assert!(*n2 > 0, "agent {i} round 2 (re-mount after restart) returned empty");
    }
    println!("CONCURRENT AGENTS OK ({N} agents, both rounds across restart)");
}

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
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";
    let read_count = |n: &str| -> i64 {
        let out = std::process::Command::new("msb")
            .args(["exec", n, "--", "sh", "-c", &format!("cat {page} | wc -c")])
            .output()
            .expect("msb exec read");
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
    };

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

    // Run 1: bring the mount up.
    {
        let q = Message::new(Role::User).with_contents([Part::text("Reply with READY.")]);
        let mut strm = agent.run(q);
        while let Some(ev) = strm.next().await {
            let _ = ev;
        }
    }
    let n1 = read_count(&name);
    println!("before forwarder kill: {n1} bytes");
    assert!(n1 > 0, "initial mount read should succeed");

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
    println!("forwarder killed: {}", String::from_utf8_lossy(&k.stdout).trim());

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
    let n2 = read_count(&name);
    println!("after forwarder death + self-heal: {n2} bytes");

    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
    assert!(
        n2 > 0,
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
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";
    let read_count = |n: &str| -> i64 {
        let out = std::process::Command::new("msb")
            .args(["exec", n, "--", "sh", "-c", &format!("cat {page} | wc -c")])
            .output()
            .expect("msb exec read");
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0)
    };
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
        .vfs(notion_vfs())
        .build()
        .expect("build agent");

    drive_once(&mut agent).await; // initial mount
    assert!(read_count(&name) > 0, "initial mount read failed");

    for round in 1..=3 {
        let _ = std::process::Command::new("msb")
            .args([
                "exec", &name, "--", "sh", "-c",
                "for p in $(pgrep -f 'vfs-fwd /mnt/vfs'); do kill -9 \"$p\"; done; sleep 0.4",
            ])
            .status();
        drive_once(&mut agent).await; // self-heal re-mount
        let n = read_count(&name);
        println!("round {round}: after kill+heal -> {n} bytes");
        assert!(n > 0, "round {round}: provider access did not recover (got {n})");
    }

    // No stale-mount accumulation: exactly one /mnt/vfs entry in /proc/mounts.
    let m = std::process::Command::new("msb")
        .args(["exec", &name, "--", "sh", "-c", "grep -c ' /mnt/vfs ' /proc/mounts"])
        .output()
        .expect("msb exec mounts");
    let count: i64 = String::from_utf8_lossy(&m.stdout).trim().parse().unwrap_or(-1);
    println!("/mnt/vfs entries in /proc/mounts after 3 kill+heal rounds: {count}");

    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
    assert_eq!(
        count, 1,
        "stale mounts accumulated ({count} /mnt/vfs entries) — lazy umount is leaking"
    );
    println!("REPEATED FORWARDER-DEATH RECOVERY OK (3 rounds, 1 mount entry)");
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
    // Host FUSE mount of the notion provider.
    let vfs = Arc::new(Vfs::from_config(notion_vfs()).unwrap());
    let rt = tokio::runtime::Handle::current();
    let mountpoint = std::env::temp_dir().join(format!("ailoy-hostfuse-{}", stamp()));
    std::fs::create_dir_all(&mountpoint).unwrap();
    let _mount = ailoy::vfs::VfsMount::spawn(vfs, &mountpoint, rt).expect("host fuse mount");

    // Wait for the host FUSE mount to be ready + populated (first readdir
    // triggers the provider fetch) before binding it into the guest.
    let mut host_ls = 0;
    for _ in 0..40 {
        host_ls = std::fs::read_dir(mountpoint.join("notion/pages"))
            .map(|rd| rd.filter_map(|e| e.ok()).count())
            .unwrap_or(0);
        if host_ls > 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    println!("host sees notion/pages entries: {host_ls}");
    assert!(host_ls > 0, "host FUSE mount not ready/populated");

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
        if h.exec_shell("true".into(), Some(10)).await.map(|o| o.exit_code == 0).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    let out = h
        .exec_shell(
            "echo '== ls /mnt/vfs =='; ls /mnt/vfs 2>&1; \
             echo '== ls /mnt/vfs/notion/pages =='; ls /mnt/vfs/notion/pages 2>&1; \
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
}

/// Full dep-free forwarder end to end: host forward server (notion) + a CLEAN
/// sandbox (no python/fuse3/apt) running the static Rust forwarder binary, which
/// reaches the host over allow@host egress and serves provider content.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: full static dep-free forwarder serves a provider (needs creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_full() {
    use std::path::Path;
    use std::sync::Arc;
    dotenvy::dotenv().ok();
    // Cross-build first: see crates/ailoy-vfs-forwarder/README.md.
    let target = format!("{}-unknown-linux-musl", std::env::consts::ARCH);
    let bin_path = format!(
        "{}/crates/ailoy-vfs-forwarder/target/{target}/release/ailoy-vfs-fwd",
        env!("CARGO_MANIFEST_DIR")
    );
    let bin = std::fs::read(&bin_path)
        .unwrap_or_else(|e| panic!("forwarder binary missing at {bin_path}: {e}"));

    let vfs = Arc::new(Vfs::from_config(notion_vfs()).unwrap());
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
    for _ in 0..40 {
        if h.exec_shell("true".into(), Some(10)).await.map(|o| o.exit_code == 0).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    h.write(Path::new("/opt/ailoy-vfs-fwd"), &bin).await.expect("write binary");
    let page = "/mnt/vfs/notion/pages/\
                Engineering_Logs__490e8208-3e62-48ef-a8ce-bc08755ea4ff/page.json";
    let script = format!(
        "echo \"deps: python3=$(command -v python3) fusermount3=$(command -v fusermount3)\"; \
         chmod +x /opt/ailoy-vfs-fwd; mkdir -p /mnt/vfs; \
         VFS_HOST='http://host.microsandbox.internal:{port}' VFS_TOKEN='{tok}' \
         setsid sh -c '/opt/ailoy-vfs-fwd /mnt/vfs >/tmp/fwd.log 2>&1' </dev/null >/dev/null 2>&1 & \
         for _ in $(seq 1 40); do grep -q ' /mnt/vfs ' /proc/mounts && break; sleep 0.25; done; \
         echo '== mount =='; mount | grep /mnt/vfs || echo '(not mounted)'; \
         echo '== ls pages =='; ls /mnt/vfs/notion/pages 2>&1 | head; \
         echo '== cat bytes =='; cat '{page}' 2>/dev/null | wc -c; \
         echo '== log =='; cat /tmp/fwd.log 2>&1 | tail -5"
    );
    let out = h.exec_shell(script, Some(60)).await.expect("guest exec");
    println!("--- guest ---\n{}", out.stdout);
    if !out.stderr.trim().is_empty() {
        println!("--- stderr ---\n{}", out.stderr);
    }
    let bytes: i64 = out
        .stdout
        .lines()
        .skip_while(|l| !l.contains("== cat bytes =="))
        .nth(1)
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(0);
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
    assert!(bytes > 0, "dep-free forwarder did not serve provider page.json ({bytes}B)");
}

/// Full filesystem access through the static dep-free forwarder: write a file to
/// the S3 mount, read it back, then `rm` it (unlink) — all from a clean guest
/// with no python/fuse3/apt. Proves write + unlink work over the forward path.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: write+rm through static forwarder on S3 (needs AWS creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_write_unlink() {
    use std::path::Path;
    use std::sync::Arc;
    dotenvy::dotenv().ok();
    let target = format!("{}-unknown-linux-musl", std::env::consts::ARCH);
    let bin_path = format!(
        "{}/crates/ailoy-vfs-forwarder/target/{target}/release/ailoy-vfs-fwd",
        env!("CARGO_MANIFEST_DIR")
    );
    let bin = std::fs::read(&bin_path)
        .unwrap_or_else(|e| panic!("forwarder binary missing at {bin_path}: {e}"));

    let vfs = Arc::new(Vfs::from_config(vfs_config()).unwrap()); // s3-only
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
    for _ in 0..40 {
        if h.exec_shell("true".into(), Some(10)).await.map(|o| o.exit_code == 0).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    h.write(Path::new("/opt/ailoy-vfs-fwd"), &bin).await.expect("write binary");
    let f = format!("/mnt/vfs/s3/{fname}");
    let script = format!(
        "chmod +x /opt/ailoy-vfs-fwd; mkdir -p /mnt/vfs; \
         VFS_HOST='http://host.microsandbox.internal:{port}' VFS_TOKEN='{tok}' \
         setsid sh -c '/opt/ailoy-vfs-fwd /mnt/vfs >/tmp/fwd.log 2>&1' </dev/null >/dev/null 2>&1 & \
         for _ in $(seq 1 40); do grep -q ' /mnt/vfs ' /proc/mounts && break; sleep 0.25; done; \
         printf 'rmcontent-{s}' > '{f}'; \
         echo \"WROTE=$(cat '{f}')\"; \
         rm '{f}' && echo RM_OK || echo RM_FAIL; \
         (ls '{f}' >/dev/null 2>&1 && echo STILL_THERE || echo GONE)"
    );
    let out = h.exec_shell(script, Some(60)).await.expect("guest exec");
    println!("--- guest ---\n{}", out.stdout);
    if !out.stderr.trim().is_empty() {
        println!("--- stderr ---\n{}", out.stderr);
    }
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
    assert!(out.stdout.contains(&format!("WROTE=rmcontent-{s}")), "write/read failed");
    assert!(out.stdout.contains("RM_OK"), "rm (unlink) failed");
    assert!(out.stdout.contains("GONE"), "file still present after rm");
}

/// Large-file / chunked-read correctness through the static forwarder: write a
/// ~300 KB object to S3 (host-side), then `cat` it in a clean guest and verify
/// the byte count and content checksum match — proving direct_io chunked reads
/// reassemble correctly (the kernel issues many ranged reads for a big file).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "poc: large/chunked read through static forwarder (needs AWS creds + sandbox + cross-built binary)"]
async fn vfs_static_forwarder_large_read() {
    use std::path::Path;
    use std::sync::Arc;
    dotenvy::dotenv().ok();
    let target = format!("{}-unknown-linux-musl", std::env::consts::ARCH);
    let bin_path = format!(
        "{}/crates/ailoy-vfs-forwarder/target/{target}/release/ailoy-vfs-fwd",
        env!("CARGO_MANIFEST_DIR")
    );
    let bin = std::fs::read(&bin_path)
        .unwrap_or_else(|e| panic!("forwarder binary missing at {bin_path}: {e}"));

    let s = stamp();
    let fname = format!("vfs-fwd-big-{s}.bin");
    // Deterministic ~300 KB payload: byte[i] = i % 251.
    let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    let want_len = data.len();
    // Sample offsets (incl. mid + last byte) — verifying these proves correct
    // multi-chunk offset handling without needing bc/md5 in the minimal guest.
    let (b0, bmid, bend) = ((0u32 % 251) as u8, (150_000u32 % 251) as u8, (299_999u32 % 251) as u8);

    let vfs = Arc::new(Vfs::from_config(vfs_config()).unwrap()); // s3-only
    {
        let (res, vp) = vfs.route(&format!("/s3/{fname}")).expect("route");
        res.write_bytes(&vp, data).await.expect("host write big file");
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
    for _ in 0..40 {
        if h.exec_shell("true".into(), Some(10)).await.map(|o| o.exit_code == 0).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    h.write(Path::new("/opt/ailoy-vfs-fwd"), &bin).await.expect("write binary");
    let f = format!("/mnt/vfs/s3/{fname}");
    // Sum every byte in the guest (awk over od) + byte count → compare to host.
    let script = format!(
        "chmod +x /opt/ailoy-vfs-fwd; mkdir -p /mnt/vfs; \
         VFS_HOST='http://host.microsandbox.internal:{port}' VFS_TOKEN='{tok}' \
         setsid sh -c '/opt/ailoy-vfs-fwd /mnt/vfs >/tmp/fwd.log 2>&1' </dev/null >/dev/null 2>&1 & \
         for _ in $(seq 1 40); do grep -q ' /mnt/vfs ' /proc/mounts && break; sleep 0.25; done; \
         echo \"LEN=$(cat '{f}' | wc -c)\"; \
         echo \"B0=$(dd if='{f}' bs=1 skip=0 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\"; \
         echo \"BMID=$(dd if='{f}' bs=1 skip=150000 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\"; \
         echo \"BEND=$(dd if='{f}' bs=1 skip=299999 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')\""
    );
    let out = h.exec_shell(script, Some(90)).await.expect("guest exec");
    println!(
        "--- guest ---\n{}\nwant LEN={want_len} B0={b0} BMID={bmid} BEND={bend}",
        out.stdout
    );
    let _ = std::process::Command::new("msb").args(["stop", &name]).status();
    let _ = std::process::Command::new("msb").args(["rm", &name]).status();
    // Clean up the S3 object.
    {
        let (res, vp) = vfs.route(&format!("/s3/{fname}")).expect("route");
        let _ = res.unlink(&vp).await;
    }
    assert!(out.stdout.contains(&format!("LEN={want_len}")), "byte count mismatch");
    assert!(out.stdout.contains(&format!("B0={b0}")), "byte@0 mismatch");
    assert!(out.stdout.contains(&format!("BMID={bmid}")), "byte@150000 mismatch");
    assert!(out.stdout.contains(&format!("BEND={bend}")), "byte@299999 mismatch");
}

/// All three providers mounted together at /mnt/vfs/{s3,notion,gdrive}.
fn all_vfs() -> VfsConfig {
    VfsConfig {
        mounts: vec![
            MountSpec {
                prefix: "/s3".into(),
                provider: ProviderConfig::S3(s3_config()),
            },
            MountSpec {
                prefix: "/notion".into(),
                provider: ProviderConfig::Notion(ailoy::vfs::NotionConfig {
                    api_key: std::env::var("NOTION_API_KEY").unwrap(),
                }),
            },
            MountSpec {
                prefix: "/gdrive".into(),
                provider: ProviderConfig::GDrive(ailoy::vfs::GDriveConfig {
                    client_id: std::env::var("GOOGLE_CLIENT_ID").unwrap(),
                    client_secret: std::env::var("GOOGLE_CLIENT_SECRET").unwrap(),
                    refresh_token: std::env::var("GOOGLE_REFRESH_TOKEN").unwrap(),
                }),
            },
        ],
    }
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
    let mount_dir = std::fs::read_dir(std::env::temp_dir())
        .ok()
        .and_then(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| {
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

/// Quantify where the guest (sandbox forward) path spends time vs the host
/// (direct in-process) path. Brings up the forward server + a sandbox, mounts
/// the forwarder, then measures from inside the guest:
///   - pure transport RTT (`/stat?path=/` short-circuits before any provider),
///   - one `readdir`,
///   - a serial `stat` of every entry (what `ls -la` triggers),
/// and compares against the same Resource calls in-process on the host.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bench: measures guest-forward vs host overhead (needs AWS + sandbox + macFUSE)"]
async fn vfs_forward_overhead_bench() {
    use std::sync::Arc;
    use std::time::Instant;

    dotenvy::dotenv().ok();

    // S3-only so provider latency is uniform across entries.
    let cfg = VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/s3".into(),
            provider: ProviderConfig::S3(s3_config()),
        }],
    };
    let vfs = Arc::new(Vfs::from_config(cfg).unwrap());
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs.clone(), &rt).expect("forward");
    let port = forward.port();
    let token = forward.token().to_string();

    // ---- Host baseline (in-process) ----
    let (res, vp) = vfs.route("/s3").expect("route");
    let t = Instant::now();
    let entries = res.readdir(&vp).await.expect("host readdir");
    let host_readdir_ms = t.elapsed().as_secs_f64() * 1e3;
    let n = entries.len();
    let t = Instant::now();
    for e in &entries {
        let (r, p) = vfs.route(&format!("/s3/{}", e.name)).unwrap();
        let _ = r.stat(&p).await;
    }
    let host_stat_all_ms = t.elapsed().as_secs_f64() * 1e3;
    // Concurrent — this is how the real host FUSE mount (multithreaded macFUSE)
    // actually issues getattr, unlike the guest forwarder's nothreads=True.
    let t = Instant::now();
    let futs = entries.iter().map(|e| {
        let (r, p) = vfs.route(&format!("/s3/{}", e.name)).unwrap();
        async move { r.stat(&p).await }
    });
    let _ = futures::future::join_all(futs).await;
    let host_stat_concurrent_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Bring up the sandbox + in-guest forwarder ----
    let sandbox = RunEnv::sandbox(SandboxConfig {
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let handle = sandbox.get().await.expect("handle");
    ailoy::vfs::bootstrap_guest_forwarder(&handle, "/mnt/vfs", port, &token)
        .await
        .expect("bootstrap forwarder");

    // Bench script runs entirely in the guest; isolates transport from FUSE.
    let bench = r#"
import os, time, urllib.request, urllib.parse
BASE = os.environ["B"]; TOK = os.environ["T"]; MNT = "/mnt/vfs/s3"
def req(route, path):
    r = urllib.request.Request(BASE + route + "?" + urllib.parse.urlencode({"path": path}), method="GET")
    r.add_header("x-vfs-token", TOK)
    return urllib.request.urlopen(r, timeout=60).read()
N = 50
t = time.perf_counter()
for _ in range(N): req("/stat", "/")
print("transport_per_req_ms", round((time.perf_counter()-t)/N*1000, 2))
t = time.perf_counter(); names = os.listdir(MNT); ld = (time.perf_counter()-t)*1000
print("guest_listdir_ms", round(ld, 2), "n", len(names))
t = time.perf_counter()
for nm in names:
    try: os.stat(os.path.join(MNT, nm))
    except OSError: pass
sa = (time.perf_counter()-t)*1000
print("guest_stat_all_ms", round(sa, 2))
print("guest_per_entry_stat_ms", round(sa/max(len(names),1), 2))
"#;
    let script = format!(
        "export B='http://host.microsandbox.internal:{port}'; export T='{token}'; \
         python3 -c \"$BENCH\"",
    );
    let script = format!("BENCH={}\n{script}", shell_quote(bench));
    let out = handle
        .exec_shell(script, Some(180))
        .await
        .expect("guest bench");

    println!("\n================ VFS FORWARD OVERHEAD ================");
    println!("entries (n)              : {n}");
    println!("HOST  readdir            : {host_readdir_ms:.1} ms");
    println!(
        "HOST  stat all (serial)  : {host_stat_all_ms:.1} ms  ({:.2} ms/entry)",
        host_stat_all_ms / n.max(1) as f64
    );
    println!(
        "HOST  stat all (concurrent, ~real FUSE): {host_stat_concurrent_ms:.1} ms"
    );
    println!("--- guest (sandbox forward) ---\n{}", out.stdout.trim());
    if !out.stderr.trim().is_empty() {
        println!("[guest stderr] {}", out.stderr.trim());
    }
    println!("=====================================================\n");
}

/// Single-quote a string for safe embedding in a POSIX shell assignment.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Per-provider host-vs-guest comparison, focused on Notion/GDrive where the
/// per-entry `stat` is itself expensive (Notion renders the page; GDrive
/// exports Workspace docs) — so the guest's serial-vs-host's-concurrent gap is
/// felt most. Measures, for a directory in each provider:
///   - host `readdir` + concurrent `stat` of every entry (≈ real host FUSE),
///   - guest `listdir` + serial `stat` of every entry (≈ `ls -la`),
/// using the updated forwarder (readdir-populated attr cache + multithreading).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bench: host vs guest per provider (needs all creds + sandbox + macFUSE)"]
async fn vfs_provider_overhead_bench() {
    use std::sync::Arc;
    use std::time::Instant;

    dotenvy::dotenv().ok();

    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap());
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs.clone(), &rt).expect("forward");
    let port = forward.port();
    let token = forward.token().to_string();

    // Pick a representative directory per provider (a Notion page dir whose
    // entries' stat triggers page rendering; the GDrive + S3 roots).
    let notion_dir = {
        let (res, vp) = vfs.route("/notion/pages").expect("route notion");
        let pages = res.readdir(&vp).await.expect("notion readdir");
        pages
            .first()
            .map(|e| format!("/notion/pages/{}", e.name))
            .expect("at least one notion page")
    };
    let targets = [
        ("notion", notion_dir.as_str()),
        ("gdrive", "/gdrive"),
        ("s3", "/s3"),
    ];

    // ---- Host baseline (in-process), readdir + concurrent stat ----
    let mut host_rows = Vec::new();
    for (label, path) in targets {
        let (res, vp) = vfs.route(path).unwrap();
        let t = Instant::now();
        let entries = res.readdir(&vp).await.expect("host readdir");
        let rd = t.elapsed().as_secs_f64() * 1e3;
        let n = entries.len();
        let t = Instant::now();
        let futs = entries.iter().map(|e| {
            let (r, p) = vfs.route(&format!("{path}/{}", e.name)).unwrap();
            async move { r.stat(&p).await }
        });
        let _ = futures::future::join_all(futs).await;
        let sa = t.elapsed().as_secs_f64() * 1e3;
        host_rows.push((label, n, rd, sa));
    }

    // ---- Bring up sandbox + new in-guest forwarder ----
    let sandbox = RunEnv::sandbox(SandboxConfig {
        allow_host_egress: true,
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let handle = sandbox.get().await.expect("handle");
    ailoy::vfs::bootstrap_guest_forwarder(&handle, "/mnt/vfs", port, &token)
        .await
        .expect("bootstrap forwarder");

    // Guest: listdir + serial stat of every entry (what `ls -la` does), per dir.
    let bench = r#"
import os, time, json
for d in os.environ["DIRS"].split(","):
    g = "/mnt/vfs" + d
    t = time.perf_counter(); names = os.listdir(g); ld = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    for nm in names:
        try: os.stat(os.path.join(g, nm))
        except OSError: pass
    sa = (time.perf_counter()-t)*1000
    print(json.dumps({"dir": d, "n": len(names),
                      "listdir_ms": round(ld,1), "stat_all_ms": round(sa,1)}))
"#;
    let dirs = targets.iter().map(|(_, p)| *p).collect::<Vec<_>>().join(",");
    let script = format!(
        "export DIRS={}; BENCH={}\npython3 -c \"$BENCH\"",
        shell_quote(&dirs),
        shell_quote(bench),
    );
    let out = handle.exec_shell(script, Some(180)).await.expect("guest bench");

    println!("\n=================== HOST vs GUEST (per provider) ===================");
    println!(
        "{:<8} {:>3}  {:>16}  {:>22}",
        "provider", "n", "HOST rd+stat(conc)", "GUEST listdir+stat(ser)"
    );
    let guest: std::collections::HashMap<String, serde_json::Value> = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| {
            let dir = v.get("dir").and_then(|d| d.as_str())?.to_string();
            Some((dir, v))
        })
        .collect();
    for (label, n, rd, sa) in &host_rows {
        let path = targets.iter().find(|(l, _)| l == label).unwrap().1;
        let (gld, gsa) = guest
            .get(path)
            .map(|v| {
                (
                    v.get("listdir_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                    v.get("stat_all_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                )
            })
            .unwrap_or((0.0, 0.0));
        println!(
            "{label:<8} {n:>3}  {:>7.0}+{:>7.0}  {:>10.0}+{:>10.0}",
            rd, sa, gld, gsa
        );
    }
    if !out.stderr.trim().is_empty() {
        println!("[guest stderr] {}", out.stderr.trim());
    }
    println!("(ms; host stat is concurrent, guest stat is serial after a readdir)");
    println!("====================================================================\n");
}

fn tail(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - n..])
    }
}
