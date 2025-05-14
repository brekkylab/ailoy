import mcp
import pytest

from ailoy import AsyncRuntime
from ailoy.agent import Agent

from ..common import print_agent_response


@pytest.mark.asyncio
async def test_mcp_tools_github():
    rt = AsyncRuntime("inproc://")
    ex = Agent(rt, model_name="qwen3-8b")
    await ex.initialize()

    await ex.add_tools_from_mcp_server(
        mcp.StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ),
        tools_to_add=[
            "search_repositories",
            "get_file_contents",
        ],
    )
    tool_names = set([tool.desc.name for tool in ex._tools])
    assert "search_repositories" in tool_names
    assert "get_file_contents" in tool_names

    query = "Search the repository named brekkylab/ailoy, and summarize its README.md."
    async for resp in ex.run(query):
        print_agent_response(resp)

    await ex.deinitialize()
