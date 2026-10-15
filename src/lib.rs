// Lets proc-macro expansions use the `ailoy::` prefix inside this crate.
extern crate self as ailoy;

pub mod agent;
pub mod console;
pub mod datatype;
mod macros;
pub mod memory;
pub mod message;
pub mod model;
pub mod tool;

/// For [`to_value!`], whose expansion names `IndexMap` in the caller's crate, which need not
/// depend on `indexmap` itself.
#[doc(hidden)]
pub use indexmap as __indexmap;

/// A started console on virtx's default server, for tests.
///
/// Fetches the server when `$VIRTX_HOME/bin` has none, names an image because a session boots
/// on one, and mounts the host's temp dir at its own path so a test can hand the console a
/// `tempfile` it wrote and read the result back. Panics on failure: a console that cannot start
/// should stop the test run loudly.
#[cfg(test)]
pub(crate) async fn test_console() -> virtx::console::ConsoleClient {
    dotenvy::dotenv().ok();

    virtx::ensure_virtx()
        .await
        .unwrap_or_else(|e| panic!("fetching the console server: {e:#}"));
    let tmp = std::env::temp_dir();
    let mut console = virtx::console::ConsoleClient::builder()
        .image(virtx::image::Recipe::new("python:3.12-slim"))
        .mount(tmp.clone(), tmp)
        .build()
        .await
        .unwrap_or_else(|e| panic!("starting the console server: {e:#}"));
    console.start().await.expect("starting a test console");
    console
}
