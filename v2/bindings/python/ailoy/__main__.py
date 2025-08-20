import asyncio
from ailoy import Part, Message, LocalLanguageModel

async def main():
    user_message = Message("user")
    user_message.append_content(Part(part_type="text", text="Show me the money."))

    model = None
    async for v in LocalLanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v.comment, v.current, v.total)
        if v.result:
            model = v.result
    
    async for resp in model.run([user_message]):
        print(resp)

if __name__ == "__main__":
    asyncio.run(main())
