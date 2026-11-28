//! Where an agent runs: the run loop ([`agent`]), the models it calls ([`model`]), the tools it
//! makes ([`tool`]) and the sandbox it works in ([`console`]). An [`AgentProvider`] holds what
//! an `ailoy_core::agent::AgentSpec` only names, and the agents kept by name, each with its
//! sessions. It is served over HTTP by two routers: [`router`], to `ailoy_core::Client`, and
//! [`admin_router`], to `ailoy_core::AdminClient`.
//!
//! The data an agent is made of and says, [`datatype`] and [`message`], is `ailoy-core`'s.

pub mod agent;
pub mod console;
mod macros;
pub mod model;
mod provider;
mod router;
pub mod tool;

pub use ailoy_core::{datatype, message, to_value, tool_func};
pub use provider::{AgentProvider, Refusal};
pub use router::{admin_router, router};

/// A sandbox of its own, for tests, on one virtx server per test process.
///
/// The server is virtx's default, `$VIRTX_STDIO` when set, and `.env` is read for it; the image
/// is pulled once. Panics on failure: a sandbox that cannot be made should stop the test run
/// loudly.
#[cfg(test)]
pub(crate) async fn test_sandbox() -> virtx::Sandbox {
    const IMAGE: &str = "python:3.12-slim";
    static CLIENT: tokio::sync::OnceCell<virtx::Client> = tokio::sync::OnceCell::const_new();
    let client = CLIENT
        .get_or_init(|| async {
            dotenvy::dotenv().ok();
            let client = virtx::Client::spawn_virtx([] as [&str; 0])
                .await
                .unwrap_or_else(|e| panic!("starting the virtx server: {e}"));
            client
                .pull_image(IMAGE, None)
                .complete(false)
                .await
                .unwrap_or_else(|e| panic!("pulling {IMAGE}: {e}"));
            client
        })
        .await;
    let mut args = virtx::SandboxCreateArgs::new(IMAGE);
    args.memory_mib = Some(1024);
    client
        .create_sandbox(&args)
        .await
        .unwrap_or_else(|e| panic!("making a test sandbox: {e}"))
}
