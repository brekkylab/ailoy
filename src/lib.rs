// Lets proc-macro expansions use the `ailoy::` prefix inside this crate.
extern crate self as ailoy;

pub mod agent;
pub mod automation;
pub mod console;
pub mod datatype;
pub mod lang_model;
mod macros;
pub mod memory;
pub mod message;
pub mod tool;

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
