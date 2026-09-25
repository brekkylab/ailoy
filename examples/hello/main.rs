//! Run one agent turn with a fixed question: no tools, no console, no system message.
//!
//! ```sh
//! cargo run --example hello -- [model]
//! ```
//!
//! `model` defaults to `openai/gpt-5.4-mini`; its provider's API key has to be set
//! (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), in the environment or in `.env`.

use std::io::Write as _;

use ailoy::{
    agent::{Agent, AgentSpec},
    message::{Message, Part, Role},
};
use futures::StreamExt as _;

/// The request the agent is given.
const QUERY: &str = "What is the meaning of hello world?";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ailoy loads `.env` only under `#[cfg(test)]`, so a binary has to.
    dotenvy::dotenv().ok();

    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "openai/gpt-5.4-mini".to_string());
    let mut agent = Agent::try_new(AgentSpec::new(&model)).await?;
    println!("model  {model}\n");

    let query = Message::new(Role::User).with_contents([Part::text(QUERY)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let output = output?;
        let message = &output.message;
        if message.role == Role::Assistant {
            for text in message.contents.iter().filter_map(Part::as_text) {
                println!("{text}");
            }
        }
        std::io::stdout().flush()?;
    }
    Ok(())
}
