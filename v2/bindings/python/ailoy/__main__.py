import asyncio
import ailoy._core as ailoy_core

async def main():
    print(dir(ailoy_core))
    msg = ailoy_core.Message("user")
    msg.push_content(ailoy_core.Part(type="text", text="Hello world"))
    print(msg)
    async for v in ailoy_core.LanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v)

if __name__ == "__main__":
    asyncio.run(main())
