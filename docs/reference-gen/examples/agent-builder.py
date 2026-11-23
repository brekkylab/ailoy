import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    agent = await (
        AgentBuilder("anthropic/claude-sonnet-5-5")
        .instruction("You are a research assistant. Name the pages you read.")
        .web_search_tool(["DuckDuckGo", "Bing"])
        .web_fetch_tool()
        .thinking_effort("medium")
        .context_manager(max_input_tokens=100_000)
        .build()
    )
    async with agent:
        async for out in agent.run("What changed in the latest Python release?"):
            for part in out["message"]["contents"]:
                if part["type"] == "text":
                    print(part["text"])


asyncio.run(main())
