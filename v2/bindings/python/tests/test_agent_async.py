import asyncio
from typing import Literal

import pytest
import pytest_asyncio

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def agent():
    model = await ai.LocalLanguageModel.create(
        "Qwen/Qwen3-0.6B", progress_callback=lambda prog: print(prog)
    )
    model.disable_reasoning()
    agent = ai.Agent(model)
    return agent


async def test_simple_chat(agent: ai.Agent):
    agg = ai.MessageAggregator()
    async for resp in agent.run("What is your name?"):
        message = agg.update(resp)
        if message:
            print(message)


async def test_builtin_tool(agent: ai.Agent):
    tool = ai.BuiltinTool.terminal()
    agent.add_tool(tool)
    agg = ai.MessageAggregator()
    async for resp in agent.run("List the files in the current directory."):
        message = agg.update(resp)
        if message:
            print(message)


async def test_python_async_function_tool(agent: ai.Agent):
    async def tool_temperature(
        location: str, unit: Literal["Celsius", "Fahrenheit"] = "Celsius"
    ):
        """
        Get temperature of the provided location
        Args:
            location: The city name
            unit: The unit of temperature
        Returns:
            int: The temperature
        """
        await asyncio.sleep(1.0)
        return 36

    tool = ai.PythonAsyncFunctionTool(func=tool_temperature)

    agent.add_tool(tool)
    agg = ai.MessageAggregator()
    async for resp in agent.run("What is the temperature in Seoul now?"):
        message = agg.update(resp)
        if message:
            print(message)

    agent.remove_tool(tool.description.name)


async def test_mcp_tools(agent: ai.Agent):
    tools = await ai.MCPTransport.Stdio("uvx", ["mcp-server-time"]).tools("time")
    agent.add_tools(tools)
    agg = ai.MessageAggregator()
    async for resp in agent.run("What time is it now in Asia/Seoul?"):
        message = agg.update(resp)
        if message:
            print(message)

    agent.remove_tools([t.description.name for t in tools])
