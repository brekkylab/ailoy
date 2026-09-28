//! Events: appended to the `events` table with a type, and the background tasks that
//! produce them for a trigger's own sources. Nothing here filters or decides.

use std::{collections::HashMap, path::Path, sync::Arc};

use chrono::Utc;
use notify::{RecursiveMode, Watcher};
use serde_json::{Value, json};
use tokio::{sync::Notify, task::JoinHandle};

use crate::{
    db::{Db, now},
    trigger::{SourceConfig, TriggerConfig},
};

/// Type of the events carrying a trigger's run results.
pub fn run_type(trigger: &str) -> String {
    format!("runs:{trigger}")
}

pub fn parse_cron(schedule: &str) -> anyhow::Result<croner::Cron> {
    croner::Cron::new(schedule)
        .with_seconds_optional()
        .parse()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Appends an event and wakes the dispatcher.
#[derive(Clone)]
pub struct Publisher {
    pub db: Db,
    pub wake: Arc<Notify>,
}

impl Publisher {
    pub fn publish(&self, kind: &str, payload: Value) -> anyhow::Result<i64> {
        let (id, dirtied) = self.db.publish(kind, &payload, now())?;
        if dirtied > 0 {
            self.wake.notify_one();
        }
        Ok(id)
    }
}

/// The background tasks behind every trigger's own sources, keyed by trigger.
pub struct SourceTasks {
    publisher: Publisher,
    tasks: HashMap<String, Vec<JoinHandle<()>>>,
}

impl SourceTasks {
    pub fn new(publisher: Publisher) -> Self {
        Self {
            publisher,
            tasks: HashMap::new(),
        }
    }

    /// (Re)start the tasks a trigger's sources need.
    pub fn start(&mut self, trigger: &str, config: &TriggerConfig) {
        self.stop(trigger);
        let mut handles = Vec::new();
        for (kind, source) in &config.sources {
            let publisher = self.publisher.clone();
            let kind = kind.clone();
            let handle = match source {
                SourceConfig::Cron { schedule } => {
                    let Ok(cron) = parse_cron(schedule) else {
                        continue;
                    };
                    tokio::spawn(async move {
                        loop {
                            let next = match cron.find_next_occurrence(&Utc::now(), false) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!(kind, "cron: {e}");
                                    return;
                                }
                            };
                            let wait = (next - Utc::now()).to_std().unwrap_or_default();
                            tokio::time::sleep(wait).await;
                            if let Err(e) =
                                publisher.publish(&kind, json!({ "at": next.timestamp() }))
                            {
                                tracing::error!(kind, "publish: {e}");
                            }
                        }
                    })
                }
                SourceConfig::Fs { path, events } => {
                    let path = path.clone();
                    let wanted = events.clone();
                    tokio::spawn(async move {
                        if let Err(e) = watch_fs(&publisher, &kind, &path, &wanted).await {
                            tracing::error!(kind, "fs watch: {e}");
                        }
                    })
                }
            };
            handles.push(handle);
        }
        if !handles.is_empty() {
            self.tasks.insert(trigger.to_string(), handles);
        }
    }

    pub fn stop(&mut self, trigger: &str) {
        for h in self.tasks.remove(trigger).unwrap_or_default() {
            h.abort();
        }
    }
}

async fn watch_fs(
    publisher: &Publisher,
    kind: &str,
    path: &Path,
    wanted: &[String],
) -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<notify::Event>(256);
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let _ = tx.blocking_send(event);
        }
    })?;
    watcher.watch(path, RecursiveMode::Recursive)?;
    while let Some(event) = rx.recv().await {
        let change = match event.kind {
            notify::EventKind::Create(_) => "create",
            notify::EventKind::Modify(_) => "modify",
            notify::EventKind::Remove(_) => "remove",
            _ => continue,
        };
        if !wanted.is_empty() && !wanted.iter().any(|w| w == change) {
            continue;
        }
        for p in event.paths {
            publisher.publish(
                kind,
                json!({ "path": p.to_string_lossy(), "event": change }),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cron_five_fields_parse_and_bad_ones_do_not() {
        assert!(parse_cron("*/5 * * * *").is_ok());
        assert!(parse_cron("not a cron").is_err());
    }
}
