// Lets proc-macro expansions use the `ailoy::` prefix inside this crate.
extern crate self as ailoy;

pub mod agent;
pub mod browser;
pub mod console;
pub mod datatype;
mod macros;
pub mod message;
pub mod model;
pub mod tool;

/// For [`to_value!`], whose expansion names `IndexMap` in the caller's crate, which need not
/// depend on `indexmap` itself.
#[doc(hidden)]
pub use indexmap as __indexmap;

/// A sandbox of its own, for tests, on one virtx server per test process.
///
/// The server is virtx's default, `$VIRTX_STDIO` when set, and `.env` is read for it; the image
/// is pulled once. Panics on failure: a sandbox that cannot be made should stop the test run
/// loudly.
#[cfg(test)]
pub(crate) async fn test_sandbox() -> virtx::Sandbox {
    use futures::StreamExt as _;

    const IMAGE: &str = "python:3.12-slim";
    static CLIENT: tokio::sync::OnceCell<virtx::Client> = tokio::sync::OnceCell::const_new();
    let client = CLIENT
        .get_or_init(|| async {
            dotenvy::dotenv().ok();
            let client =
                tokio::task::spawn_blocking(|| virtx::Client::spawn_virtx([] as [&str; 0]))
                    .await
                    .unwrap()
                    .unwrap_or_else(|e| panic!("starting the virtx server: {e}"));
            let mut log = client.pull_image(IMAGE, None).log();
            while let Some(entry) = log.next().await {
                let entry = entry.unwrap_or_else(|e| panic!("pulling {IMAGE}: {e}"));
                if let virtx::ImageJobEntry::End(end) = entry.entry {
                    assert!(end.error.is_none(), "pulling {IMAGE}: {:?}", end.error);
                }
            }
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
