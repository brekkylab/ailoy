use std::sync::{Arc, Weak};

use async_trait::async_trait;
use tokio::sync::{Mutex, MutexGuard};

#[async_trait]
pub trait Container: Send + Sync + 'static {
    type Handle: Console;

    async fn boot(&mut self) -> Self::Handle;

    async fn shutdown(&mut self);
}

/// Object-safe erased view of `Container`. The blanket impl below adapts any
/// concrete `Container` by boxing the handle into `Arc<Mutex<dyn Executable>>`.
#[async_trait]
trait ContainerDyn: Send + Sync + 'static {
    async fn boot(&mut self) -> Arc<Mutex<dyn Console>>;

    async fn shutdown(&mut self);
}

#[async_trait]
impl<B: Container> ContainerDyn for B {
    async fn boot(&mut self) -> Arc<Mutex<dyn Console>> {
        // UFCS so this doesn't recurse into the ContainerDyn impl we're in.
        Arc::new(Mutex::new(Container::boot(self).await))
    }

    async fn shutdown(&mut self) {
        Container::shutdown(self).await;
    }
}

/// Execution result from a shell command.
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

#[async_trait]
pub trait Console: Send + Sync {
    /// Run `program` with `args`. `timeout` is in seconds; when elapsed the
    /// resulting [`ExecResult`] has `timed_out = true`.
    ///
    /// Takes `&self` so a single booted handle can be shared by multiple
    /// `RunenvHandle` clones. Implementations are responsible for their own
    /// internal synchronization (e.g. a Mutex around the stdin/stdout pipe).
    async fn exec(
        &mut self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult>;
}

enum RunenvState {
    Idle,
    /// Holds a `Weak` so that when the last user-visible `RunenvHandle` drops,
    /// the inner `Arc` count reaches zero and `RunenvInner::drop` fires.
    Running(Weak<RunenvHandle>),
}

#[derive(Clone)]
pub struct Runenv {
    machine: Arc<Mutex<dyn ContainerDyn>>,
    state: Arc<Mutex<RunenvState>>,
}

impl Runenv {
    pub fn new<B: Container>(machine: B) -> Self {
        Self {
            machine: Arc::new(Mutex::new(machine)),
            state: Arc::new(Mutex::new(RunenvState::Idle)),
        }
    }

    pub async fn get(&self) -> Arc<RunenvHandle> {
        loop {
            let mut s = self.state.lock().await;
            match &*s {
                RunenvState::Idle => {
                    // Hold the state lock through boot; other callers block on
                    // `state.lock().await` and observe `Running` once we finish.
                    let exec = self.machine.lock().await.boot().await;
                    let handle = Arc::new(RunenvHandle {
                        machine: self.machine.clone(),
                        exec,
                        state: self.state.clone(),
                    });
                    *s = RunenvState::Running(Arc::downgrade(&handle));
                    return handle;
                }
                RunenvState::Running(weak) => {
                    if let Some(inner) = weak.upgrade() {
                        return inner;
                    }
                    // Last user handle just dropped but the `Drop`-spawned
                    // shutdown hasn't acquired the state lock yet. Release the
                    // state lock so it can make progress, then retry.
                    drop(s);
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct RunenvHandle {
    machine: Arc<Mutex<dyn ContainerDyn>>,
    exec: Arc<Mutex<dyn Console>>,
    state: Arc<Mutex<RunenvState>>,
}

impl RunenvHandle {
    pub async fn get(&self) -> MutexGuard<'_, dyn Console> {
        self.exec.lock().await
    }
}

impl Drop for RunenvHandle {
    fn drop(&mut self) {
        let machine = self.machine.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            // Hold the state lock through shutdown so concurrent `get()` calls
            // block until we transition back to `Idle`.
            let mut s = state.lock().await;
            machine.lock().await.shutdown().await;
            *s = RunenvState::Idle;
        });
    }
}
