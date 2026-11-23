import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").web_fetch_tool().build() as agent:
        turn = agent.run("Read https://example.com and summarize it.")  # nothing has run yet
        async for out in turn:
            print(out["message"]["role"], out["finish_reason"]["type"])
            if out["message"]["role"] == "tool":
                break  # ends the turn here, after the first tool result


asyncio.run(main())
