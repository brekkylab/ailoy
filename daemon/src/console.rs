//! The one console every trigger script runs in.
//!
//! Kept booted for as long as the daemon lives, so a script call costs a process
//! spawn. Its image, network answer and packages are the union of what the registered
//! automations declare; when a registration changes the union, the console is rebuilt
//! before the next call.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use ailoy::automation::{AutomationDef, GUEST_SCRATCH};
use anyhow::Context as _;
use virtx::{
    console::ConsoleClient,
    image::{ImageSource, Recipe},
    protocol::ExecResp,
};

use crate::{config::ConsoleConfig, db::Db};

pub struct SharedConsole {
    config: ConsoleConfig,
    scratch: PathBuf,
    console: Option<ConsoleClient>,
    /// What the current console was built for.
    union_key: Option<String>,
}

/// The network answer and packages the registered automations need together.
#[derive(Debug, Default)]
struct Union {
    network: bool,
    packages: BTreeSet<String>,
}

impl Union {
    fn of(db: &Db) -> anyhow::Result<Self> {
        let mut u = Union::default();
        for t in db.list_triggers()? {
            let Ok(def) = AutomationDef::load(&t.config.automation) else {
                continue;
            };
            u.network |= def.console.network;
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
        }
    }

    pub fn is_up(&self) -> bool {
        self.console.is_some()
    }

    /// Forget the current union, so the next call rebuilds if anything changed.
    pub fn invalidate(&mut self) {
        self.union_key = None;
    }

    /// Where the scratch tree is, as a command in the console spells it.
    pub fn guest_scratch(&self) -> &Path {
        Path::new(GUEST_SCRATCH)
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

        let image = self.config.image.as_str();
        let mut builder = ConsoleClient::builder()
            .network(union.network)
            .mount(self.scratch.clone(), GUEST_SCRATCH);
        builder = if union.packages.is_empty() {
            builder.image(ImageSource::reference(image))
        } else {
            let pkgs: Vec<&str> = union.packages.iter().map(String::as_str).collect();
            builder.image(
                Recipe::new(image).step(format!("pip install --no-cache-dir {}", pkgs.join(" "))),
            )
        };
        if let Some(cmd) = &self.config.program {
            builder = builder.cmd(cmd);
        }
        let mut console = builder
            .build()
            .await
            .with_context(|| format!("console server {:?}", self.config.program))?;
        console
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("starting the trigger console: {e}"))?;
        tracing::info!(
            network = union.network,
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
            Err(virtx::protocol::Failure::Broken(e)) => {
                self.console = None;
                self.union_key = None;
                Err(anyhow::anyhow!("trigger console broke: {e}"))
            }
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    }
}
