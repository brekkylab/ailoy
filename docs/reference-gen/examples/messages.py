import asyncio

from ailoy import AgentBuilder
from ailoy.types import Message

question: Message = {
    "role": "user",
    "contents": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image", "image": {"type": "url", "url": "https://example.com/cat.png"}},
    ],
}


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        async for out in agent.run(question):
            message = out["message"]
            for call in message.get("tool_calls", []):
                print("calls", call["function"]["name"], call["function"]["arguments"])
            for part in message["contents"]:
                if part["type"] == "text":
                    print(part["text"])


asyncio.run(main())
