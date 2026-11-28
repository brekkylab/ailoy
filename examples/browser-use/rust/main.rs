//! An agent with a browser, waiting for what you ask of it.
//!
//! ```sh
//! cargo run --example browser-use
//! ```
//!
//! The agent works in a sandbox with python and curl and virtx's `browser` abin, a headless
//! Chromium, which it drives with its `browser` and `browser_screenshot` tools; the shell and
//! file tools are there too, for what is easier done without the browser, such as fetching a
//! page's data or computing over it. Each line you type is the next turn; `/exit` ends.
//!
//! `AILOY_MODEL` is the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//! has to be set up as `docs/content/guides/model-providers.mdx` says, any key in
//! the environment or in `.env`.

use ailoy::{
    agent::AgentBuilder,
    message::{Message, Part, Role},
};
use futures::StreamExt as _;
use virtx::{Client, Recipe, SandboxCreateArgs};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-browser-use:1";

/// What the agent has.
const HAS: &str = "a sandbox and a web browser in it, which you drive with the browser tools";

/// What the agent is for.
const INSTRUCTION: &str = "Do what the user asks, and say briefly what you are doing as you go. \
     When you are done, report what you found plainly, with the figures and links that matter. \
     Answer in the user's language.";

/// What to ask, for a start.
const SUGGESTIONS: &[&str] = &[
    "Solve today's New York Times Wordle puzzle.",
    "Search Skyscanner and Google Flights for round-trip weekend flights from Seoul to Tokyo next month, and list the three cheapest by price.",
    "Find three wireless mice under $50 at an online store, and compare their prices and main features.",
];

/// One line from stdin, `None` once it closes. Bytes that are not UTF-8, as a broken IME
/// composition can leave, become U+FFFD instead of failing the read.
fn read_line() -> std::io::Result<Option<String>> {
    use std::io::BufRead;
    let mut bytes = Vec::new();
    let read = std::io::stdin().lock().read_until(b'\n', &mut bytes)?;
    Ok((read > 0).then(|| String::from_utf8_lossy(&bytes).into_owned()))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ailoy loads `.env` only under `#[cfg(test)]`, so a binary has to.
    dotenvy::dotenv().ok();
    let model = std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".into());

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = Client::spawn_virtx([] as [&str; 0]).await?;

    // The image, built once: python, and curl on it.
    // The server answers a tag in its full form, as `docker.io/library/ailoy-browser-use:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        let recipe = Recipe::new("python:3.12-slim-trixie").step(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
             && rm -rf /var/lib/apt/lists/*",
        );
        client.build_image(recipe, IMAGE).complete(false).await?;
    }

    // The browser is the sandbox's: its abin, a network for the pages it opens, and the memory
    // Chromium takes.
    let mut args = SandboxCreateArgs::new(IMAGE).abin("browser");
    args.network = true;
    args.memory_mib = Some(2048);
    let sandbox = client.create_sandbox(&args).await?;

    let mut agent = AgentBuilder::new(&model)
        .instruction(format!(
            "You are a helpful assistant with {HAS}. {INSTRUCTION}"
        ))
        .sandbox(sandbox)
        .browser()
        .build()
        .await?;
    println!("model     {model}\n");

    println!("Ask anything, or try:");
    for suggestion in SUGGESTIONS {
        println!("  - {suggestion}");
    }
    println!("`/exit` ends.\n");

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;
        // A closed stdin ends the session, as `/exit` does.
        let Some(line) = tokio::task::spawn_blocking(read_line).await?? else {
            break;
        };
        let query = line.trim();
        if query == "/exit" {
            break;
        }
        if query.is_empty() {
            continue;
        }
        let message = Message::new(Role::User).with_contents([Part::text(query)]);
        let mut turn = agent.run(message);
        while let Some(output) = turn.next().await {
            let message = output?.message;
            if message.role != Role::Assistant {
                continue;
            }
            for text in message.contents.iter().filter_map(Part::as_text) {
                println!("{}", text.trim());
            }
            for call in message.tool_calls.iter().flatten() {
                if let Some((_, name, args)) = call.as_function() {
                    let args = serde_json::to_string(args)?;
                    println!("  [{name}] {}", args.chars().take(160).collect::<String>());
                }
            }
        }
        println!();
    }
    Ok(())
}
