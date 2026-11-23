import asyncio

from ailoy import AgentBuilder, register_mcp_stdio, register_tool


def get_weather(city: str) -> str:
    return f"Sunny and 21°C in {city}"


# Your own function: the desc is what the model sees, the function what runs
weather = register_tool(
    {
        "name": "get_weather",
        "description": "The current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    get_weather,
)


async def main() -> None:
    # An MCP server's tools, named fs__<tool>
    fs_tools = await register_mcp_stdio(
        "fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    agent = await AgentBuilder("anthropic/claude-sonnet-5-5").tool(weather).tools(fs_tools).build()
    async with agent:
        async for out in agent.run("What's the weather in Seoul?"):
            print(out["message"]["role"], out["message"]["contents"])


asyncio.run(main())
