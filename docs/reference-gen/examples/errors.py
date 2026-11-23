import asyncio

from ailoy import AgentBuilder, AiloyError


async def main() -> None:
    try:
        # no org called "nobody" has a model provider
        agent = await AgentBuilder("nobody/no-model").build()
    except AiloyError as e:
        print("ailoy refused:", e)
        return
    await agent.close()


asyncio.run(main())
