//! What every part of the daemon shares.

use std::{path::PathBuf, sync::Arc};

use tokio::sync::{Mutex, Notify};

use crate::{
    config::{Config, TRIGGER_DIR, WORK_DIR},
    console::SharedConsole,
    db::Db,
    sources::{Publisher, SourceTasks},
};

pub struct AppState {
    pub root: PathBuf,
    pub config: Config,
    pub db: Db,
    /// Appends events and wakes the dispatcher.
    pub publisher: Publisher,
    pub worker_wake: Arc<Notify>,
    pub sources: Mutex<SourceTasks>,
    pub console: Mutex<SharedConsole>,
    pub http: reqwest::Client,
}

impl AppState {
    pub fn new(root: PathBuf, config: Config, db: Db) -> Arc<Self> {
        let publisher = Publisher {
            db: db.clone(),
            wake: Arc::new(Notify::new()),
        };
        let console = SharedConsole::new(config.console.clone(), root.join(TRIGGER_DIR));
        Arc::new(Self {
            sources: Mutex::new(SourceTasks::new(publisher.clone())),
            console: Mutex::new(console),
            publisher,
            worker_wake: Arc::new(Notify::new()),
            http: reqwest::Client::new(),
            root,
            config,
            db,
        })
    }

    /// Runner work directory of a trigger.
    pub fn work_dir(&self, trigger: &str) -> PathBuf {
        self.root.join(WORK_DIR).join(trigger)
    }

    /// A run's directory, as the runner lays it out.
    pub fn run_dir(&self, trigger: &str, run_id: &str) -> PathBuf {
        self.work_dir(trigger)
            .join(ailoy::automation::RUNS_DIR)
            .join(run_id)
    }

    /// The trigger console's scratch subdirectory of a trigger, on the host side.
    pub fn trigger_dir(&self, trigger: &str) -> PathBuf {
        self.root.join(TRIGGER_DIR).join(trigger)
    }
}
