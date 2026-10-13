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
/// Panics on failure: a missing server binary should stop the test run loudly.
#[cfg(test)]
pub(crate) async fn test_console() -> virtx::console::ConsoleClient {
    dotenvy::dotenv().ok();

    let mut console = virtx::console::ConsoleClient::builder()
        .build()
        .await
        .unwrap_or_else(|e| panic!("starting the console server: {e:#}"));
    console.start().await.expect("starting a test console");
    console
}
