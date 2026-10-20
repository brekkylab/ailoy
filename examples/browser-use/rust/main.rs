//! An agent with a browser, waiting for what you ask of it.
//!
//! ```sh
//! cargo run --example browser-use [-- --noshell]
//! ```
//!
//! The agent uses your own browser (Chrome by default), in a window of its own opened on the ailoy
//! docs, and drives it with its `browser_*` tools. It also has a console with python and curl, for
//! what is easier done there, such as fetching a page's data or computing over it; `--noshell`
//! leaves it out, so the agent has the browser alone and needs no console server. Turn on remote
//! debugging at chrome://inspect/#remote-debugging (Chrome 144 or later) and allow the browser's
//! prompt, which appears as it starts. Each line you type is the next turn; `/exit` ends. When a
//! site needs you, to sign in say, the agent says so and ends its turn: do it in its window, then
//! tell it to go on.
//!
//! `AILOY_MODEL` is the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//! API key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …), in the environment or in
//! `.env`.

use ailoy::{
    agent::AgentBuilder,
    browser::Browser,
    message::{Message, Part, Role},
};
use futures::StreamExt as _;
use virtx::{console::ConsoleClient, image::Recipe};

/// Where the browser starts.
const START: &str = "https://brekkylab.github.io/ailoy/";

/// What the agent has, with a console and without.
const WITH_CONSOLE: &str = "a web browser and a console: the browser is the user's own, \
     which you drive with the browser tools";
const BROWSER_ONLY: &str = "a web browser: the user's own, which you drive with the \
     browser tools";

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

    let shell = !std::env::args().skip(1).any(|a| a == "--noshell");

    // The browser tools reach the browser from this process; the console is for the other tools.
    let mut builder = AgentBuilder::new(&model)
        .instruction(format!(
            "You are a helpful assistant with {}. {INSTRUCTION}",
            if shell { WITH_CONSOLE } else { BROWSER_ONLY }
        ))
        .browser(Browser::host().start_url(START));
    if shell {
        // The console server, fetched into virtx's cache the first time: a host that installed
        // only ailoy has none.
        virtx::ensure_virtx().await?;
        let console = ConsoleClient::builder()
            .image(Recipe::new("python:3.12-slim-trixie").step(
                "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
                 && rm -rf /var/lib/apt/lists/*",
            ))
            .network(true)
            .build()
            .await?;
        builder = builder.system_tools().console(console);
    }
    let mut agent = builder.build().await?;
    println!("model     {model}");
    println!(
        "has       {}\n",
        if shell { "browser, console" } else { "browser" }
    );

    println!(
        "Allow the browser's prompt to let the agent in. It opens the ailoy docs as an example."
    );
    agent.start_browser().await?;
    println!("The agent has a window of its own in your browser.\n");

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
