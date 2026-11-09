//! A `Sandbox` from `@brekkylab/virtx`, as the sandbox an agent runs in.
//!
//! Both packages reach the server through one `libvirtx_core` in the process: this addon opens
//! the one `@brekkylab/virtx` opened, by the path virtx's `libraryPath()` answers, and takes a
//! hold of the sandbox's client's `VirtxClient*` of its own, so the sandbox the agent holds is the
//! caller's, over the same session.

use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{Mutex, PoisonError},
};

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

/// The sandbox `sandbox`, a `@brekkylab/virtx` `Sandbox`, names, over its own client.
pub fn sandbox(env: &Env, sandbox: &Object<'_>) -> Result<virtx::Sandbox> {
    let not_one = || invalid("expected a Sandbox from @brekkylab/virtx");
    let get = |e: napi::Error| invalid(e.reason);
    let client = sandbox
        .get::<Object>("client")
        .map_err(get)?
        .ok_or_else(not_one)?;
    let id = sandbox
        .get::<String>("id")
        .map_err(get)?
        .ok_or_else(not_one)?;
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
    // borrows it; the sandbox takes a hold of its own.
    let borrowed = ManuallyDrop::new(
        unsafe { virtx::Client::from_raw(raw as *mut c_void) }.map_err(virtx_error)?,
    );
    let info = virtx::SandboxInfo {
        id,
        image: sandbox.get("image").map_err(get)?.ok_or_else(not_one)?,
        snapshot: sandbox.get("snapshot").map_err(get)?,
        running: sandbox.get("running").map_err(get)?.ok_or_else(not_one)?,
        ports: Vec::new(),
        abins: sandbox.get("abins").map_err(get)?.unwrap_or_default(),
    };
    Ok((*borrowed).sandbox(info))
}
