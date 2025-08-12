import asyncio
import ailoy._core as ailoy_core

async def main():
    async for v in ailoy_core.LanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v)

if __name__ == "__main__":
    asyncio.run(main())
