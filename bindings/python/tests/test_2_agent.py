import textwrap

import mcp
import pytest

from ailoy import Runtime
from ailoy.agent import Agent, AgentResponse
from ailoy.vector_store import VectorStore

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
def runtime():
    rt = Runtime("inproc://agent")
    yield rt
    rt.stop()


@pytest.fixture(scope="module")
def agent(runtime: Runtime):
    agent = Agent(runtime, model_name="qwen3-8b")
    yield agent
    agent.delete()


def test_tool_call_calculator(agent: Agent):
    agent._messages.clear()
    agent._tools.clear()
    agent.add_tools_from_preset("calculator")

    query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))"
    for resp in agent.run(query):
        _print_agent_response(resp)


def test_tool_call_frankfurter(agent: Agent):
    agent._messages.clear()
    agent._tools.clear()
    agent.add_tools_from_preset("frankfurter")

    query = "I want to buy 250 U.S. Dollar and 350 Chinese Yuan with my Korean Won. How much do I need to take?"
    for resp in agent.run(query):
        _print_agent_response(resp)


def test_mcp_tools_github(agent: Agent):
    agent._messages.clear()
    agent._tools.clear()
    agent.add_tools_from_mcp_server(
        mcp.StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ),
        tools_to_add=[
            "search_repositories",
            "get_file_contents",
        ],
    )
    tool_names = set([tool.desc.name for tool in agent._tools])
    assert "search_repositories" in tool_names
    assert "get_file_contents" in tool_names

    query = "Summarize README.md from repository brekkylab/ailoy."
    for resp in agent.run(query):
        _print_agent_response(resp)


def test_simple_rag_pipeline(runtime: Runtime, agent: Agent):
    agent._messages.clear()
    agent._tools.clear()

    with VectorStore(runtime, embedding_model_name="bge-m3") as vs:
        vs.insert(
            "Ailoy is a lightweight library for building AI applications — such as **agent systems** or **RAG pipelines** — with ease. It is designed to enable AI features effortlessly, one can just import and use.",  # noqa: E501
        )
        query = "What is Ailoy?"
        items = vs.retrieve(query)
        prompt = f"""
            Based on the following contexts, answer to user's question.
            Context: {[item.document for item in items]}
            Question: {query}
        """
        for resp in agent.run(prompt):
            _print_agent_response(resp)
