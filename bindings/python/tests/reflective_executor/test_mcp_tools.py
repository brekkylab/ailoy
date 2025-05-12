import asyncio

import mcp
import pytest

from ailoy import AsyncRuntime
from ailoy.reflective_executor import ReflectiveExecutor


@pytest.mark.asyncio
async def test_mcp_tools_github():
    rt = AsyncRuntime("inproc://")
    ex = ReflectiveExecutor(rt, model_name="qwen3-0.6b")
    await ex.initialize()

    await ex.add_tools_from_mcp_server(
        mcp.StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ),
    )
    tool_names = set([tool.desc.name for tool in ex._tools])
    # https://github.com/modelcontextprotocol/servers/tree/main/src/github#tools
    for tool_name in [
        "create_or_update_file",
        "push_files",
        "search_repositories",
        "create_repository",
        "get_file_contents",
        "create_issue",
        "create_pull_request",
        "fork_repository",
        "create_branch",
        "list_issues",
        "update_issue",
        "add_issue_comment",
        "search_code",
        "search_issues",
        "search_users",
        "list_commits",
        "get_issue",
        "get_pull_request",
        "list_pull_requests",
        "create_pull_request_review",
        "merge_pull_request",
        "get_pull_request_files",
        "get_pull_request_status",
        "update_pull_request_branch",
        "get_pull_request_comments",
        "get_pull_request_reviews",
    ]:
        assert tool_name in tool_names

    await ex.deinitialize()


if __name__ == "__main__":
    asyncio.run(test_mcp_tools_github())
