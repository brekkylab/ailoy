from pathlib import Path

import mcp
import pytest

from ailoy import Runtime, VectorStore
from ailoy.agent import Agent

pytestmark = [pytest.mark.agent]


@pytest.fixture(scope="module")
def runtime():
    with Runtime("inproc://agent") as rt:
        yield rt


@pytest.fixture(scope="module")
def agent(runtime: Runtime):
    with Agent(runtime, model_name="Qwen/Qwen3-8B") as agent:
        yield agent


def test_tool_call_calculator(agent: Agent):
    agent.clear_messages()
    agent._tools.clear()
    agent.add_tools_from_preset("calculator")

    query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))"
    for resp in agent.query(query):
        agent.print(resp)


def test_tool_call_frankfurter(agent: Agent):
    agent.clear_messages()
    agent._tools.clear()
    agent.add_tools_from_preset("frankfurter")

    query = "I want to buy 250 U.S. Dollar and 350 Chinese Yuan with my Korean Won. How much do I need to take?"
    for resp in agent.query(query):
        agent.print(resp)


def test_tool_call_py_function(agent: Agent):
    def get_current_temperature(location: str, unit: str):
        """
        Get the current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, Country"
            unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
        Returns:
            The current temperature at the specified location in the specified units, as a float.
        """
        if unit == "celsius":
            return 25
        elif unit == "fahrenheit":
            return 77
        return

    agent.clear_messages()
    agent._tools.clear()
    agent.add_py_function_tool(f=get_current_temperature)
    tool_names = set([tool.desc.name for tool in agent._tools])
    assert "get_current_temperature" in tool_names

    query = "Hello, how is the current weather in my city Seoul?"
    for resp in agent.query(query):
        agent.print(resp)


def test_mcp_tools_filesystem(agent: Agent):
    agent.clear_messages()
    agent._tools.clear()

    path = Path(__file__).parent.parent.absolute()
    agent.add_tools_from_mcp_server(
        "filesystem",
        mcp.StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(path),
            ],
        ),
    )
    tool_names = set([tool.desc.name for tool in agent._tools])
    assert "filesystem-list_directory" in tool_names
    assert "filesystem-read_file" in tool_names

    query = f"Create a file hello.txt under {path} and write 'hello world'."
    for resp in agent.query(query):
        agent.print(resp)

    assert (path / "hello.txt").exists()
    assert (path / "hello.txt").read_text() == "hello world"
    (path / "hello.txt").unlink()


def test_simple_rag_pipeline(runtime: Runtime, agent: Agent):
    agent.clear_messages()
    agent._tools.clear()

    with VectorStore(runtime, embedding_model_name="BAAI/bge-m3", vector_store_name="faiss") as vs:
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
        for resp in agent.query(prompt):
            agent.print(resp)
