import asyncio
import os

from mcp import StdioServerParameters

from ailoy import AsyncRuntime, Agent

from common import print_agent_response


async def main():
    rt = AsyncRuntime()

    github_pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", None)
    if github_pat is None:
        github_pat = input("Enter GITHUB_PERSONAL_ACCESS_TOKEN: ")

    agent = Agent(rt, model_name="qwen3-8b")

    await agent.initialize()

    await agent.add_tools_from_mcp_server(
        StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_pat},
        ),
        # You can add more tools as your need.
        # See https://github.com/modelcontextprotocol/servers/tree/main/src/github#tools for the entire tool list.
        tools_to_add=[
            "search_repositories",
            "get_file_contents",
        ],
    )

    print('Simple Github MCP Agent (Please type "exit" to stop conversation)')

    while True:
        query = input("\nUser: ")

        if query == "exit":
            break

        if query == "":
            continue

        async for resp in agent.run(query):
            print_agent_response(resp)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
