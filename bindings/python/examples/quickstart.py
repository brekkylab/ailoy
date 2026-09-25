"""One agent turn with a tool written in Python, printed as it happens.

    OPENAI_API_KEY=... uv run python examples/quickstart.py [model]

`model` defaults to `openai/gpt-5.4-mini`; its provider's API key has to be in the
environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
"""

import asyncio
import json
import sys

import ailoy

QUERY = "Is it warmer in Seoul or in Busan right now?"


# A tool is any callable. The model's arguments arrive as keyword arguments, and what it
# returns is the result the model reads.
async def temperature(city: str) -> dict:
    await asyncio.sleep(0.1)  # stands in for a real lookup
    return {"city": city, "celsius": {"Seoul": 18.5, "Busan": 21.0}.get(city)}


async def main(model: str) -> None:
    desc = ailoy.register_tool(
        {
            "name": "temperature",
            "description": "The temperature in a city now, in Celsius.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city's name"}},
                "required": ["city"],
            },
        },
        temperature,
    )

    agent = await ailoy.AgentBuilder(model).instruction("Answer in one sentence.").tool(desc).build()

    async for output in agent.run(QUERY):
        message = output["message"]
        if message["role"] == "assistant":
            for part in message["contents"]:
                if part["type"] == "text":
                    print(part["text"])
            for call in message.get("tool_calls", []):
                function = call["function"]
                print(f"→ {function['name']} {json.dumps(function['arguments'])}")
        elif message["role"] == "tool":
            for part in message["contents"]:
                print(f"← {json.dumps(part.get('value'))}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-5.4-mini"))
