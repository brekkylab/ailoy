use ailoy::agent::AgentBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new("anthropic/claude-sonnet-5-5")
        .instruction("Answer in one short paragraph.")
        .web_fetch_tool()
        .build()
        .await?;
    // agent.run(query) is a stream of each message the model and the tools produce
    Ok(())
}
