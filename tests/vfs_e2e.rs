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

use ailoy::agent::{Agent, AgentBuilder, AgentProvider};
use ailoy::lang_model::LangModelProvider;
use ailoy::message::{Message, Part, Role};
use ailoy::runenv::{RunEnv, SandboxConfig};
use ailoy::vfs::{MountSpec, ProviderConfig, S3Config, Vfs, VfsConfig};
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
    p.models.insert(MODEL.into(), LangModelProvider::anthropic(key));
    p
}

fn stamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
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
    println!("--- non-sandbox transcript tail ---\n{}", tail(&transcript, 500));
    assert!(verify_s3(&fname, &content).await, "non-sandbox write not found in S3");
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
    println!("--- sandbox transcript tail ---\n{}", tail(&transcript, 500));
    assert!(verify_s3(&fname, &content).await, "sandbox write not found in S3");
}

fn tail(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - n..])
    }
}
