//! A `Client` from `@brekkylab/virtx`, as the client this process's provider makes each agent's
//! sandbox with.
//!
//! Both packages reach the server through one `libvirtx_core` in the process: this addon opens
//! the one `@brekkylab/virtx` opened, by the path virtx's `libraryPath()` answers, and takes a
//! hold of the client's `VirtxClient*` of its own, so the sandboxes are made over the caller's
//! session.

use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{Mutex, PoisonError},
};

use ailoy::agent::default_provider;
use napi::{
    Env,
    bindgen_prelude::{Function, FunctionRef, Object},
};
use napi_derive::napi;

use crate::error::{Result, invalid, virtx as virtx_error};

/// `@brekkylab/virtx`'s `libraryPath`, which `index.js` hands over as it loads.
static LIBRARY_PATH: Mutex<Option<FunctionRef<(), Option<String>>>> = Mutex::new(None);

/// Take `@brekkylab/virtx`'s `libraryPath`, to open the library it opened; for `index.js`.
#[napi(js_name = "_useVirtx")]
pub fn use_virtx(library_path: Function<(), Option<String>>) -> napi::Result<()> {
    let held = library_path.create_ref()?;
    *LIBRARY_PATH.lock().unwrap_or_else(PoisonError::into_inner) = Some(held);
    Ok(())
}

/// The client `client`, a `@brekkylab/virtx` `Client`, holds, with a hold of its own.
fn client(env: &Env, client: &Object<'_>) -> Result<virtx::Client> {
    let not_one = || invalid("expected a Client from @brekkylab/virtx");
    let get = |e: napi::Error| invalid(e.reason);
    let as_raw = client
        .get::<Function<(), f64>>("asRaw")
        .map_err(get)?
        .ok_or_else(not_one)?;

    let library = LIBRARY_PATH
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .ok_or_else(|| invalid("@brekkylab/virtx was not handed over; load ailoy by its index.js"))?
        .borrow_back(env)
        .and_then(|library_path| library_path.call(()))
        .map_err(get)?
        .ok_or_else(|| invalid("@brekkylab/virtx has no library open"))?;
    virtx::load(&library).map_err(virtx_error)?;
    let raw = as_raw.apply(client, ()).map_err(get)? as usize;
    // Safety: virtx made it with the library just opened, and holds it for as long as this
    // borrows it; the clone is a hold of its own.
    let borrowed = ManuallyDrop::new(
        unsafe { virtx::Client::from_raw(raw as *mut c_void) }.map_err(virtx_error)?,
    );
    Ok((*borrowed).clone())
}

/// Make each agent's sandbox with `client`, a `@brekkylab/virtx` `Client`, from now on; with
/// `null`, make none, and an agent whose spec uses a sandbox is refused.
#[napi]
pub fn set_virtx(
    env: &Env,
    #[napi(ts_arg_type = "Client | null")] client: Option<Object<'_>>,
) -> Result<()> {
    let client = client.map(|given| self::client(env, &given)).transpose()?;
    let before = std::mem::replace(&mut default_provider().write().unwrap().virtx, client);
    // Let go of the one before off this thread: a last hold waits there for its session to end.
    std::thread::spawn(move || drop(before));
    Ok(())
}
