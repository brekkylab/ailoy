import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    agent = await (
        AgentBuilder("anthropic/claude-sonnet-5-5")
        .instruction("Answer in one short paragraph.")
        .web_fetch_tool()
        .build()
    )
    async with agent:
        async for out in agent.run("What is on https://example.com?"):
            for part in out["message"]["contents"]:
                if part["type"] == "text":
                    print(part["text"])


asyncio.run(main())
