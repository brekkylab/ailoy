//! A `ConsoleClient` from `@brekkylab/virtx`, as the console an agent runs in.
//!
//! That console lives in virtx's own addon, whose Rust types this one cannot see (each links
//! its own virtx). So an agent drives its session through `ConsoleClient._relay`: protocol
//! frames as `Buffer`s, answered on virtx's runtime, under the console's own lock (see
//! [`virtx::console::Relay`]). The two addons need only speak one protocol version, not be
//! built from one virtx.
//!
//! The agent holds the `ConsoleClient` (through `_relay`, bound to it), so the session lasts
//! at least as long as the agent. It stays the caller's: the agent never ends it, and `close()`
//! ends it under the agent too, whose next console call then rejects with `CONSOLE_BROKEN`.

use std::sync::Arc;

use ailoy::console::ConsoleClient;
use napi::{
    Status,
    bindgen_prelude::{Buffer, Function, Object, Promise},
    threadsafe_function::ThreadsafeFunction,
};
use tokio::sync::Mutex;
use virtx::{BoxFuture, console::Relay, protocol::Failure};

use crate::error::{Result, invalid};

/// What an agent holds its console in.
pub type Slot = Arc<Mutex<Option<ConsoleClient>>>;

/// `ConsoleClient._relay`, bound to its console. Threadsafe because JavaScript runs only on its
/// own thread, which a turn leaves free as promises iterate it; weak so an agent does not keep
/// the process alive.
type RelayFn = ThreadsafeFunction<Buffer, Promise<Buffer>, Buffer, Status, false, true>;

/// A slot holding a console attached to `console`'s session.
pub fn attach(console: &Object<'_>) -> Result<Slot> {
    let not_one = || {
        invalid(
            "expected a ConsoleClient from @brekkylab/virtx that can be shared with an agent \
             (one with `_relay`; upgrade @brekkylab/virtx if it is one)",
        )
    };
    let relay = console
        .get::<Function<'_, Buffer, Promise<Buffer>>>("_relay")
        .map_err(|_| not_one())?
        .ok_or_else(not_one)?;
    let mounts = console
        .get::<Vec<String>>("mounts")
        .map_err(|e| invalid(e.reason))?
        .unwrap_or_default();

    let relay: RelayFn = relay
        .bind(console)
        .and_then(|bound| {
            bound
                .build_threadsafe_function()
                .callee_handled::<false>()
                .weak::<true>()
                .build()
        })
        .map_err(|e| invalid(e.reason))?;

    Ok(Arc::new(Mutex::new(Some(ConsoleClient::attach(
        JsRelay(relay),
        mounts,
    )))))
}

struct JsRelay(RelayFn);

impl Relay for JsRelay {
    fn relay(&mut self, frame: Vec<u8>) -> BoxFuture<'_, std::result::Result<Vec<u8>, Failure>> {
        Box::pin(async move {
            let answer = self.0.call_async_catch(Buffer::from(frame)).await;
            match answer {
                Ok(promise) => promise.await.map(|frame| frame.to_vec()).map_err(broken),
                Err(e) => Err(broken(e)),
            }
        })
    }
}

/// A rejection's own message. It comes back wrapped once more, as
/// `"GenericFailure, <message>"`, which is napi's bookkeeping rather than virtx's.
fn broken(error: napi::Error) -> Failure {
    let reason = error.reason.as_str();
    let reason = reason.strip_prefix("GenericFailure, ").unwrap_or(reason);
    Failure::Broken(anyhow::Error::msg(reason.to_string()))
}
