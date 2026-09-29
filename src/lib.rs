// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod console;
pub mod datatype;
pub mod lang_model;
mod macros;
pub mod memory;
pub mod message;
pub mod tool;

/// A started console, for the tests across this crate that need one.
///
/// Test scaffolding, on the console server cortex starts by default.
///
/// Panics rather than returning an error: a test with no console is meaningless, so
/// a missing server binary should stop the run and say so.
#[cfg(test)]
pub(crate) async fn test_console() -> cortex::console::ConsoleClient {
    dotenvy::dotenv().ok();

    let mut console = cortex::console::ConsoleClient::builder()
        .build()
        .await
        .unwrap_or_else(|e| panic!("starting the console server: {e:#}"));
    console.start().await.expect("starting a test console");
    console
}
