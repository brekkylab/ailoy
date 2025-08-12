import asyncio
from ailoy import Part, Message, LocalLanguageModel

async def main():
    msg = Message("user")
    msg.append_content(Part(type="text", text="Hello world"))
    msg.reasoning = [Part(type="text", text="Thinking")]
    print(msg.content)
    print(msg)
    async for v in LocalLanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v)

if __name__ == "__main__":
    asyncio.run(main())
