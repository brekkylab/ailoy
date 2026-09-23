// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod console;
pub mod datatype;
pub mod lang_model;
pub(crate) mod macros;
pub mod memory;
pub mod message;
pub mod tool;
pub(crate) mod util;

/// A started host-local console, for the tests across this crate that need one.
///
/// Test scaffolding. Which console server to start is the caller's decision
/// everywhere else in this crate; here the caller is the test suite, and it picks
/// cortex's `Backend::local` — the server cortex builds and carries, so there is
/// no program to install. `$CORTEX_LOCAL_CONSOLE_BIN` runs another build of it.
///
/// Panics rather than returning an error: a test with no console is meaningless, so
/// a server that will not start should stop the run and say so.
#[cfg(test)]
pub(crate) async fn test_console() -> cortex::console::Console {
    dotenvy::dotenv().ok();

    let mut console = cortex::console::Console::builder()
        .backend(cortex::console::Backend::local())
        .build()
        .await
        .unwrap_or_else(|e| panic!("starting the local console: {e:#}"));
    console.start().await.expect("starting a test console");
    console
}
