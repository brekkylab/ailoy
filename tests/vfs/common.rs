//! Shared helpers and provider configs for the vfs integration tests.
#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};

pub use ailoy::{
    agent::{Agent, AgentBuilder, AgentProvider},
    lang_model::LangModelProvider,
    message::{Message, Part, Role},
    runenv::{RunEnv, SandboxConfig},
    vfs::{FileKind, MountSpec, ProviderConfig, S3Config, Vfs, VfsConfig},
};
pub use futures::StreamExt;

pub const MODEL: &str = "anthropic/claude-haiku-4-5";

pub fn s3_config() -> S3Config {
    S3Config {
        bucket: std::env::var("AWS_S3_BUCKET").unwrap(),
        region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        endpoint: None,
        key_prefix: None,
    }
}

pub fn vfs_config() -> VfsConfig {
    VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/s3".into(),
            provider: ProviderConfig::S3(s3_config()),
        }],
    }
}

pub fn provider() -> AgentProvider {
    let key = std::env::var("ANTHROPIC_API_KEY").unwrap();
    let mut p = AgentProvider::new();
    p.models
        .insert(MODEL.into(), LangModelProvider::anthropic(key));
    p
}

pub fn stamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub async fn drive(mut agent: Agent, task: &str) -> String {
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

pub async fn verify_s3(fname: &str, want: &str) -> bool {
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

pub fn task_for(fname: &str, content: &str) -> String {
    format!(
        "Your instructions list an external S3 mount path. \
         First run `ls` on that s3 mount directory. \
         Then create a file named `{fname}` in that s3 mount directory whose only \
         content is the exact text `{content}`, using a shell redirect. \
         Then `cat` that file to confirm. Report what you did concisely."
    )
}

pub fn notion_vfs() -> VfsConfig {
    VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/notion".into(),
            provider: ProviderConfig::Notion(ailoy::vfs::NotionConfig {
                api_key: std::env::var("NOTION_API_KEY").unwrap(),
            }),
        }],
    }
}

pub fn gdrive_vfs() -> VfsConfig {
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

/// All three providers mounted together at /mnt/vfs/{s3,notion,gdrive}.
pub fn all_vfs() -> VfsConfig {
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

pub fn tail(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - n..])
    }
}
