import asyncio

from ailoy import LangModel
from ailoy.types import Message


async def main() -> None:
    model = await LangModel.create("anthropic/claude-sonnet-5-5")
    messages: list[Message] = [
        {"role": "system", "contents": [{"type": "text", "text": "Reply in French."}]},
        {"role": "user", "contents": [{"type": "text", "text": "Good morning!"}]},
    ]
    out = await model.infer(messages, options={"thinking_effort": "low"})
    for part in out["message"]["contents"]:
        if part["type"] == "text":
            print(part["text"])


asyncio.run(main())
