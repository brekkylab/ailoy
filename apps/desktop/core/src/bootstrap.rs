//! What this machine needs before a chat can run, fetched once when the engine starts.
//!
//! A first start has none of it: the console server and the Linux image a run's shell boots
//! on are downloads, and so is the model list. Each is a step the window shows, and a run
//! is refused until every one is done — a chat started halfway would sit on a download the
//! user cannot see, or fail on a model the picker could not list.
//!
//! Steps run in chains: a chain's steps in order, since the image is pulled by the server
//! the step before it fetched, and the chains side by side, since the model list waits on
//! neither. A chain stops at its first failure and leaves the rest of it pending;
//! [`Bootstrap::run`] again picks every chain up where it stopped.

use std::sync::Arc;

use futures::future::{BoxFuture, join_all};
use serde::Serialize;
use tokio::sync::{Mutex, watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepId {
    /// `virtx-uvm`, the console server, into virtx's cache.
    ConsoleServer,
    /// The image a run's VM boots on, pulled and built by that server.
    ConsoleImage,
    /// The model list, from models.dev.
    Catalog,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StepState {
    /// Not started: waiting on the step before it, or on a retry after that one failed.
    Pending,
    /// Since `started_at`, Unix ms.
    Running {
        started_at: i64,
    },
    Done,
    /// Not needed by this engine (no console, no fetching), and as good as done.
    Skipped,
    Failed {
        message: String,
    },
}

impl StepState {
    fn finished(&self) -> bool {
        matches!(self, StepState::Done | StepState::Skipped)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BootstrapStep {
    pub id: StepId,
    #[serde(flatten)]
    pub state: StepState,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BootstrapStatus {
    pub steps: Vec<BootstrapStep>,
    /// Every step done or skipped: a run may start.
    pub ready: bool,
}

/// One step's work. A factory rather than a future, so a failed step can be run again.
pub type Task = Arc<dyn Fn() -> BoxFuture<'static, anyhow::Result<()>> + Send + Sync>;

/// A step and its work; `None` work is a step this engine skips.
pub type Step = (StepId, Option<Task>);

pub struct Bootstrap {
    chains: Vec<Vec<Step>>,
    status: watch::Sender<BootstrapStatus>,
    /// One pass at a time: a Retry clicked while the first pass runs is that pass.
    running: Mutex<()>,
}

impl Bootstrap {
    pub fn new(chains: Vec<Vec<Step>>) -> Arc<Self> {
        let steps = chains
            .iter()
            .flatten()
            .map(|(id, task)| BootstrapStep {
                id: *id,
                state: if task.is_some() {
                    StepState::Pending
                } else {
                    StepState::Skipped
                },
            })
            .collect();
        Arc::new(Bootstrap {
            chains,
            status: watch::channel(with_ready(steps)).0,
            running: Mutex::new(()),
        })
    }

    pub fn status(&self) -> BootstrapStatus {
        self.status.borrow().clone()
    }

    /// Every change to [`status`](Self::status), for the Tauri layer to pass on.
    pub fn subscribe(&self) -> watch::Receiver<BootstrapStatus> {
        self.status.subscribe()
    }

    pub fn is_ready(&self) -> bool {
        self.status.borrow().ready
    }

    /// Run every step not yet finished. Returns at once when a pass is already running.
    pub async fn run(&self) {
        let Ok(_one) = self.running.try_lock() else {
            return;
        };
        join_all(self.chains.iter().map(|chain| self.run_chain(chain))).await;
    }

    async fn run_chain(&self, chain: &[Step]) {
        for (id, task) in chain {
            let Some(task) = task else { continue };
            if self.state_of(*id).is_some_and(|s| s.finished()) {
                continue;
            }
            self.set(
                *id,
                StepState::Running {
                    started_at: chrono::Utc::now().timestamp_millis(),
                },
            );
            match task().await {
                Ok(()) => self.set(*id, StepState::Done),
                Err(e) => {
                    tracing::warn!("bootstrap {id:?} failed: {e:#}");
                    self.set(
                        *id,
                        StepState::Failed {
                            message: format!("{e:#}"),
                        },
                    );
                    return;
                }
            }
        }
    }

    /// Resolves once every step is finished (`true`), or once a pass ends with one failed
    /// (`false`). For callers that would otherwise poll, the tests among them.
    pub async fn wait(&self) -> bool {
        let mut rx = self.subscribe();
        let settled = rx
            .wait_for(|s| {
                s.ready
                    || s.steps
                        .iter()
                        .any(|step| matches!(step.state, StepState::Failed { .. }))
            })
            .await;
        settled.is_ok_and(|s| s.ready)
    }

    /// What is still being waited on, by the names the refusal says.
    pub fn unfinished(&self) -> Vec<StepId> {
        self.status
            .borrow()
            .steps
            .iter()
            .filter(|s| !s.state.finished())
            .map(|s| s.id)
            .collect()
    }

    fn state_of(&self, id: StepId) -> Option<StepState> {
        self.status
            .borrow()
            .steps
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.state.clone())
    }

    fn set(&self, id: StepId, state: StepState) {
        self.status.send_modify(|status| {
            if let Some(step) = status.steps.iter_mut().find(|s| s.id == id) {
                step.state = state;
            }
            status.ready = status.steps.iter().all(|s| s.state.finished());
        });
    }
}

fn with_ready(steps: Vec<BootstrapStep>) -> BootstrapStatus {
    let ready = steps.iter().all(|s| s.state.finished());
    BootstrapStatus { steps, ready }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn ok() -> Task {
        Arc::new(|| Box::pin(async { Ok(()) }))
    }

    /// Fails the first `n` times it runs, then succeeds.
    fn fails(n: usize) -> (Task, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let task: Task = Arc::new(move || {
            let call = seen.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if call < n {
                    anyhow::bail!("offline")
                }
                Ok(())
            })
        });
        (task, calls)
    }

    fn state(b: &Bootstrap, id: StepId) -> StepState {
        b.state_of(id).unwrap()
    }

    #[tokio::test]
    async fn an_engine_with_nothing_to_fetch_is_ready_before_it_runs() {
        let b = Bootstrap::new(vec![
            vec![(StepId::ConsoleServer, None), (StepId::ConsoleImage, None)],
            vec![(StepId::Catalog, None)],
        ]);
        assert!(b.is_ready());
        assert_eq!(state(&b, StepId::Catalog), StepState::Skipped);
    }

    #[tokio::test]
    async fn every_chain_runs_and_then_it_is_ready() {
        let b = Bootstrap::new(vec![
            vec![
                (StepId::ConsoleServer, Some(ok())),
                (StepId::ConsoleImage, Some(ok())),
            ],
            vec![(StepId::Catalog, Some(ok()))],
        ]);
        assert!(!b.is_ready());
        assert_eq!(b.unfinished().len(), 3);
        b.run().await;
        assert!(b.is_ready());
        assert!(b.wait().await);
    }

    /// A failed server leaves the image pending rather than tried on nothing, the model list
    /// is fetched all the same, and a second pass runs only what is left.
    #[tokio::test]
    async fn a_failure_stops_its_chain_and_a_retry_resumes_it() {
        let (server, server_calls) = fails(1);
        let (catalog, catalog_calls) = fails(0);
        let b = Bootstrap::new(vec![
            vec![
                (StepId::ConsoleServer, Some(server)),
                (StepId::ConsoleImage, Some(ok())),
            ],
            vec![(StepId::Catalog, Some(catalog))],
        ]);

        b.run().await;
        assert!(
            matches!(state(&b, StepId::ConsoleServer), StepState::Failed { message } if message == "offline")
        );
        assert_eq!(state(&b, StepId::ConsoleImage), StepState::Pending);
        assert_eq!(state(&b, StepId::Catalog), StepState::Done);
        assert!(!b.wait().await, "a failed pass settles as not ready");
        assert_eq!(
            b.unfinished(),
            vec![StepId::ConsoleServer, StepId::ConsoleImage]
        );

        b.run().await;
        assert!(b.is_ready());
        assert_eq!(server_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            catalog_calls.load(Ordering::SeqCst),
            1,
            "a finished step is not run again"
        );
    }

    /// The shape the webview reads: the state as a tag beside the id, with its fields.
    #[test]
    fn a_step_serializes_flat() {
        let step = BootstrapStep {
            id: StepId::ConsoleImage,
            state: StepState::Failed {
                message: "no network".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(step).unwrap(),
            serde_json::json!({ "id": "console_image", "state": "failed", "message": "no network" })
        );
    }
}
