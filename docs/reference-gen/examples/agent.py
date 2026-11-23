import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        # run(): each whole message, once it's complete
        async for out in agent.run("Name three prime numbers."):
            for part in out["message"]["contents"]:
                if part["type"] == "text":
                    print(part["text"])

        # run_stream(): the model's words as they're written. This turn sees the last one.
        async for delta in agent.run_stream("Now add them up."):
            for piece in delta["delta"]["contents"]:
                print(piece.get("text", ""), end="", flush=True)
        print()

        print(len(agent.history), "messages so far")


asyncio.run(main())
