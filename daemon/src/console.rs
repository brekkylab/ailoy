//! The one console every trigger script runs in.
//!
//! Kept booted for as long as the daemon lives, so a script call costs a process
//! spawn. Its image, reach and secrets are the union of what the registered
//! automations declare; when a registration changes the union, the console is
//! rebuilt before the next call.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use ailoy::automation::{AutomationDef, NetworkReach};
use anyhow::Context as _;
use cortex::{
    console::{Console, ExecResp, NetworkAccess, SecretAccess},
    rootfs::Rootfs,
};

use crate::{config::ConsoleConfig, db::Db};

pub struct SharedConsole {
    config: ConsoleConfig,
    scratch: PathBuf,
    console: Option<Console>,
    /// What the current console was built for.
    union_key: Option<String>,
    guest_scratch: PathBuf,
}

/// The reach, secrets and packages the registered automations need together.
#[derive(Debug, Default)]
struct Union {
    reach: NetworkReach,
    secrets: Vec<SecretAccess>,
    packages: BTreeSet<String>,
}

fn rank(reach: NetworkReach) -> u8 {
    match reach {
        NetworkReach::None => 0,
        NetworkReach::Host => 1,
        NetworkReach::Public => 2,
        NetworkReach::Full => 3,
    }
}

impl Union {
    fn of(db: &Db) -> anyhow::Result<Self> {
        let mut u = Union::default();
        for t in db.list_triggers()? {
            let Ok(def) = AutomationDef::load(&t.config.automation) else {
                continue;
            };
            if rank(def.console.network) > rank(u.reach) {
                u.reach = def.console.network;
            }
            for s in &def.console.secrets {
                if !u.secrets.contains(s) {
                    u.secrets.push(s.clone());
                }
            }
            u.packages.extend(def.workflow.packages());
        }
        Ok(u)
    }

    fn key(&self) -> String {
        format!("{:?}", self)
    }
}

impl SharedConsole {
    pub fn new(config: ConsoleConfig, scratch: PathBuf) -> Self {
        Self {
            config,
            scratch,
            console: None,
            union_key: None,
            guest_scratch: PathBuf::new(),
        }
    }

    pub fn is_up(&self) -> bool {
        self.console.is_some()
    }

    /// Forget the current union, so the next call rebuilds if anything changed.
    pub fn invalidate(&mut self) {
        self.union_key = None;
    }

    /// Where the scratch tree is, as a command in the console spells it. Valid after
    /// [`ensure`](Self::ensure).
    pub fn guest_scratch(&self) -> &Path {
        &self.guest_scratch
    }

    /// Have a console up that matches what the registered automations need.
    pub async fn ensure(&mut self, db: &Db) -> anyhow::Result<()> {
        let union = Union::of(db)?;
        let key = union.key();
        if self.console.is_some() && self.union_key.as_deref() == Some(&key) {
            return Ok(());
        }
        if let Some(mut old) = self.console.take() {
            let _ = old.stop().await;
        }
        std::fs::create_dir_all(&self.scratch)?;

        let builder = if self.config.host {
            Console::builder()
        } else {
            let b = Console::builder()
                .network(NetworkAccess::new(union.reach.as_str()))
                .secrets(union.secrets.iter().cloned());
            if union.packages.is_empty() {
                b.image(self.config.image.as_str())
            } else {
                let pkgs: Vec<&str> = union.packages.iter().map(String::as_str).collect();
                b.rootfs(
                    Rootfs::from_image(self.config.image.as_str())
                        .run(format!("pip install --no-cache-dir {}", pkgs.join(" "))),
                )
            }
        };
        let mut console = builder
            .stdio_client(&self.config.program)
            .scratch(self.scratch.clone())
            .build()
            .await
            .with_context(|| format!("console program {:?}", self.config.program))?;
        self.guest_scratch = console
            .scratch_path()
            .ok_or_else(|| anyhow::anyhow!("the console server mounted no scratch tree"))?
            .to_path_buf();
        console
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("starting the trigger console: {e}"))?;
        tracing::info!(
            reach = union.reach.as_str(),
            packages = union.packages.len(),
            "trigger console up"
        );
        self.console = Some(console);
        self.union_key = Some(key);
        Ok(())
    }

    /// Run one command. A broken channel drops the console so the next call rebuilds.
    pub async fn exec(
        &mut self,
        argv: impl IntoIterator<Item = impl AsRef<str>>,
        timeout_ms: u64,
    ) -> anyhow::Result<ExecResp> {
        let console = self
            .console
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("trigger console is not up"))?;
        match console.exec(argv, Some(timeout_ms)).await {
            Ok(resp) => Ok(resp),
            Err(cortex::console::Failure::Broken(e)) => {
                self.console = None;
                self.union_key = None;
                Err(anyhow::anyhow!("trigger console broke: {e}"))
            }
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    }
}
