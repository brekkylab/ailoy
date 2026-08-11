// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod console;
pub mod datatype;
pub mod lang_model;
pub(crate) mod macros;
pub mod message;
pub mod tool;
pub(crate) mod util;

/// A started host-local console, for the tests across this crate that need one.
///
/// Test scaffolding. Which console server to start is the caller's decision
/// everywhere else in this crate; here the caller is the test suite, and it picks
/// `cortex-local-console` — overridable with `$AILOY_CORTEX_CONSOLE`, since the
/// binary lives in a sibling checkout rather than on `PATH` during development.
///
/// Panics rather than returning an error: a test with no console is meaningless, so
/// a missing server binary should stop the run and say so.
#[cfg(test)]
pub(crate) async fn test_console() -> cortex::console::Console {
    dotenvy::dotenv().ok();

    let program = std::env::var("AILOY_CORTEX_CONSOLE")
        .unwrap_or_else(|_| "cortex-local-console".to_string());

    let mut console = cortex::console::Console::builder()
        .stdio_client(&[&program])
        .build()
        .await
        .unwrap_or_else(|e| panic!("starting `{program}`: {e:#}"));
    console.start().await.expect("starting a test console");
    console
}
