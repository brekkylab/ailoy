//! A `virtx.ConsoleClient` from the `virtx` package, as the console an agent runs in.
//!
//! That console lives in virtx's own extension module, whose Rust types this one cannot see
//! (each links its own virtx). So an agent drives its session through
//! `ConsoleClient._relay`: protocol frames as bytes, answered on virtx's runtime, under the
//! console's own lock (see [`virtx::console::Relay`]). The two modules need only speak one
//! protocol version, not be built from one virtx.
//!
//! The agent holds a reference to the `ConsoleClient`, so the session lasts at least as long
//! as the agent. It stays the caller's: the agent never ends it, and `close()` ends it under
//! the agent too, whose next console call then raises `ConsoleBroken`.

use std::sync::Arc;

use ailoy::console::ConsoleClient;
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyBytes};
use tokio::sync::{Mutex, oneshot};
use virtx::{BoxFuture, console::Relay, protocol::Failure};

/// What an agent holds its console in.
pub type Slot = Arc<Mutex<Option<ConsoleClient>>>;

/// A slot holding a console attached to `console`'s session.
pub fn attach(console: &Bound<'_, PyAny>) -> PyResult<Slot> {
    let py = console.py();
    let class = py.import("virtx")?.getattr("ConsoleClient")?;
    if !console.is_instance(&class)? {
        return Err(PyTypeError::new_err(format!(
            "expected a virtx.ConsoleClient, got {}",
            console.get_type().name()?
        )));
    }
    if !console.hasattr("_relay")? {
        return Err(PyTypeError::new_err(
            "this virtx.ConsoleClient cannot be shared with an agent; upgrade the virtx package",
        ));
    }

    let mounts: Vec<String> = console.getattr("mounts")?.extract()?;
    let relay = PyRelay(console.clone().unbind());
    Ok(Arc::new(Mutex::new(Some(ConsoleClient::attach(
        relay, mounts,
    )))))
}

/// Frames to a `virtx.ConsoleClient`, through its `_relay`.
struct PyRelay(Py<PyAny>);

impl Relay for PyRelay {
    fn relay(&mut self, frame: Vec<u8>) -> BoxFuture<'_, Result<Vec<u8>, Failure>> {
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            // `_relay` only schedules the call, so the GIL is held for no longer than that.
            Python::attach(|py| {
                let reply = Reply(std::sync::Mutex::new(Some(tx)));
                self.0
                    .call_method1(py, "_relay", (PyBytes::new(py, &frame), reply))
            })
            .map_err(|e| broken(format!("relaying to the virtx console: {e}")))?;

            match rx.await {
                Ok(answer) => answer.map_err(broken),
                Err(_) => Err(broken("the virtx console dropped a relayed call")),
            }
        })
    }
}

fn broken(reason: impl Into<String>) -> Failure {
    Failure::Broken(anyhow::Error::msg(reason.into()))
}

/// An answer frame, or why there is none.
type Answer = oneshot::Sender<Result<Vec<u8>, String>>;

/// The callback `_relay` answers through: `(frame, None)`, or `(None, reason)` for a broken
/// channel or a closed console. Only the first call counts.
#[pyclass(frozen)]
struct Reply(std::sync::Mutex<Option<Answer>>);

#[pymethods]
impl Reply {
    fn __call__(&self, frame: Option<&[u8]>, broken: Option<String>) {
        let Some(tx) = self.0.lock().unwrap().take() else {
            return;
        };
        let answer = match frame {
            Some(frame) => Ok(frame.to_vec()),
            None => Err(broken.unwrap_or_else(|| "the virtx console answered nothing".into())),
        };
        let _ = tx.send(answer);
    }
}
