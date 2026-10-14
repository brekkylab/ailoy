//! An agent with a browser, waiting for what you ask of it.
//!
//! ```sh
//! cargo run --example browser-use [-- --host]
//! ```
//!
//! The browser starts before the first request: Chromium in a VM of its own, which you watch
//! and use in yours at the address printed, opened on the ailoy docs. Each line you type is the
//! next turn; `/exit` ends. When a site needs you, to sign in say, the agent says so and ends its
//! turn: do it in the viewer, then tell it to go on.
//!
//! With `--host` it uses your own Chrome instead, in a window of its own: turn on remote
//! debugging at chrome://inspect/#remote-debugging (Chrome 144 or later) and allow Chrome's
//! prompt, which appears as it starts.
//!
//! `AILOY_MODEL` is the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//! API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …), in the environment or in
//! `.env`.

use std::sync::Arc;

use ailoy::{
    agent::AgentBuilder,
    browser::Browser,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use futures::StreamExt as _;
use tokio::sync::Mutex;
use virtx::image::Recipe;

/// Where the browser starts.
const START: &str = "https://brekkylab.github.io/ailoy/";

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
    let host = std::env::args().skip(1).any(|a| a == "--host");

    // The console server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none.
    virtx::ensure_virtx().await?;
    // The browser is driven from this console over the network, so it has to have one.
    let console = ConsoleClient::builder()
        .image(Recipe::new("debian:trixie-slim"))
        .network(true)
        .build()
        .await?;
    // Shared, so this app can drive the agent's browser from it too.
    let console = Arc::new(Mutex::new(Some(console)));
    let browser = if host {
        Browser::host()
    } else {
        Browser::sandboxed()
    };
    let mut agent = AgentBuilder::new(&model)
        .system_tools()
        .shared_console(console.clone())
        .browser(browser)
        .build()
        .await?;
    println!("model  {model}\n");

    if host {
        println!("Allow Chrome's prompt to let the agent in.");
    }
    match agent.start_browser().await? {
        Some(url) => println!("Watch and use the browser at {url}\n"),
        None => println!("The agent has a window of its own in your Chrome.\n"),
    }

    println!("Opening the ailoy docs as an example.\n");
    let opened = console
        .lock()
        .await
        .as_mut()
        .expect("the agent keeps its console")
        .exec(["agent-browser", "open", START], None)
        .await?;
    if opened.code != 0 {
        println!(
            "Could not open {START}: {}\n",
            String::from_utf8_lossy(&opened.stderr).trim()
        );
    }

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
