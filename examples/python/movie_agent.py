import asyncio
import os

from ailoy import AsyncRuntime, Agent
from ailoy.agent import BearerAuthenticator

from common import print_reflective_response


async def main():
    rt = AsyncRuntime()

    tmdb_api_key = os.environ.get("TMDB_API_KEY", None)
    if tmdb_api_key is None:
        tmdb_api_key = input("Enter TMDB API Key: ")

    agent = Agent(rt, model_name="qwen3-8b")

    await agent.initialize()

    agent.add_tools_from_preset("tmdb", authenticator=BearerAuthenticator(tmdb_api_key))

    print('Simple Github assistant (Please type "exit" to stop conversation)')

    while True:
        query = input("\nUser: ")

        if query == "exit":
            break

        if query == "":
            continue

        async for resp in agent.run(query):
            print_reflective_response(resp)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
