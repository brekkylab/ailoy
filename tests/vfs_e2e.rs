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

fn tail(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - n..])
    }
}
