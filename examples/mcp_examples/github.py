import asyncio
import os

import ailoy as ai
from aioconsole import ainput


async def main():
    github_pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", None)
    if github_pat is None:
        github_pat = input("Enter GITHUB_PERSONAL_ACCESS_TOKEN: ")

    model = await ai.LangModel.new_local("Qwen/Qwen3-8B")
    agent = ai.Agent(model)

    os.environ.setdefault("GITHUB_PERSONAL_ACCESS_TOKEN", github_pat)
    os.environ.setdefault("GITHUB_TOOLSETS", "repos")
    mcp_client = await ai.MCPClient.from_stdio(
        "docker",
        [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
    )
    agent.add_tools(
        [
            mcp_client.get_tool("search_repositories"),
            mcp_client.get_tool("get_file_contents"),
        ]
    )

    print('Ailoy Github MCP Agent (Please type "exit" to stop conversation)')

    try:
        while True:
            query = await ainput("\n\nUser: ")

            if query == "exit":
                break

            if query == "":
                continue

            async for resp in agent.run_delta(query):
                for content in resp.delta.contents:
                    if isinstance(content, ai.PartDelta.Text):
                        print(content.text, end="", flush=True)
                    elif isinstance(content, ai.PartDelta.Value):
                        print(content.value)
                    else:
                        raise ValueError(
                            f"Content has invalid part_type: {content.part_type}"
                        )
                for tool_call in resp.delta.tool_calls:
                    if isinstance(tool_call, ai.PartDelta.Function) and isinstance(
                        tool_call.function, ai.PartDeltaFunction.Verbatim
                    ):
                        print(tool_call.function.text, end="", flush=True)
                    else:
                        raise ValueError(
                            f"Tool call has invalid part_type: {tool_call.part_type}"
                        )

    except asyncio.exceptions.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
