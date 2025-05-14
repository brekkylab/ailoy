import textwrap

import mcp
import pytest

from ailoy import Runtime
from ailoy.agent import Agent, AgentResponse

pytestmark = [pytest.mark.agent]


def _print_agent_response(resp: AgentResponse):
    if resp.type == "output_text":
        print(resp.content, end="")
        if resp.end_of_turn:
            print("\n\n")
    elif resp.type == "reasoning":
        print(f"\033[93m{resp.content}\033[0m", end="")
        if resp.end_of_turn:
            print("\n\n")
    elif resp.type == "tool_call":
        print(
            textwrap.dedent(
                f"""
                Tool Call
                - ID: {resp.content.id}
                - name: {resp.content.function.name}
                - arguments: {resp.content.function.arguments}
                """
            )
        )
    elif resp.type == "tool_call_result":
        print(
            textwrap.dedent(
                f"""
                Tool Call Result
                - ID: {resp.content.tool_call_id}
                - name: {resp.content.name}
                - Result: {resp.content.content}
                """
            )
        )


@pytest.fixture(scope="module")
def prepared_agent():
    rt = Runtime("inproc://agent")
    agent = Agent(rt, model_name="qwen3-8b")
    agent.initialize()
    yield agent
    agent.deinitialize()


def test_tool_call_calculator(prepared_agent: Agent):
    prepared_agent._messages.clear()
    prepared_agent._tools.clear()
    prepared_agent.add_tools_from_preset("calculator")

    query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))"
    for resp in prepared_agent.run(query):
        _print_agent_response(resp)


def test_tool_call_frankfurter(prepared_agent: Agent):
    prepared_agent._messages.clear()
    prepared_agent._tools.clear()
    prepared_agent.add_tools_from_preset("frankfurter")

    query = "I want to buy 250 U.S. Dollar and 350 Chinese Yuan with my Korean Won. How much do I need to take?"
    for resp in prepared_agent.run(query):
        _print_agent_response(resp)


def test_mcp_tools_github(prepared_agent: Agent):
    prepared_agent._messages.clear()
    prepared_agent._tools.clear()
    prepared_agent.add_tools_from_mcp_server(
        mcp.StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ),
        tools_to_add=[
            "search_repositories",
            "get_file_contents",
        ],
    )
    tool_names = set([tool.desc.name for tool in prepared_agent._tools])
    assert "search_repositories" in tool_names
    assert "get_file_contents" in tool_names

    query = "Summarize README.md from repository brekkylab/ailoy."
    for resp in prepared_agent.run(query):
        _print_agent_response(resp)
