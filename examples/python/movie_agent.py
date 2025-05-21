import asyncio
import os

from ailoy import Runtime, Agent
from ailoy.agent import BearerAuthenticator


async def main():
    rt = Runtime()

    tmdb_api_key = os.environ.get("TMDB_API_KEY", None)
    if tmdb_api_key is None:
        tmdb_api_key = input("Enter TMDB API Key: ")

    agent = Agent(rt, model_name="qwen3-8b")

    agent.add_tools_from_preset("tmdb", authenticator=BearerAuthenticator(tmdb_api_key))

    print('Simple movie assistant (Please type "exit" to stop conversation)')

    while True:
        query = input("\nUser: ")

        if query == "exit":
            break

        if query == "":
            continue

        for resp in agent.query(query):
            agent.print(resp)

    agent.delete()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
