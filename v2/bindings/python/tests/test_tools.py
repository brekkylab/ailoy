import asyncio
from typing import Literal

import pytest

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


# async def test_builtin_tool(agent: ai.Agent):
#     tool = ai.BuiltinTool.terminal()
#     agent.add_tool(tool)
#     agg = ai.MessageAggregator()
#     async for resp in agent.run("List the files in the current directory."):
#         message = agg.update(resp)
#         if message:
#             print(message)


async def test_python_async_function_tool():
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
        await asyncio.sleep(0.2)
        return 35 if unit == "Celsius" else 95

    def tool_temperature_sync(
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
        return 35 if unit == "Celsius" else 95

    # with async function
    tool = ai.Tool.new_py_function(tool_temperature)

    # async calls with __call__()
    assert await tool(location="Seoul") == 35
    assert await tool(location="Seoul", unit="Fahrenheit") == 95

    # async calls with call()
    assert await tool.call(**{"location": "Seoul"}) == 35
    assert await tool.call(**{"location": "Seoul", "unit": "Fahrenheit"}) == 95

    # with sync function
    tool_sync = ai.Tool.new_py_function(tool_temperature_sync)

    # async calls with __call__()
    assert await tool_sync(location="Seoul") == 35
    assert await tool_sync(location="Seoul", unit="Fahrenheit") == 95

    # async calls with call()
    assert await tool_sync.call(**{"location": "Seoul"}) == 35
    assert await tool_sync.call(**{"location": "Seoul", "unit": "Fahrenheit"}) == 95

    # sync calls with call_sync()
    assert tool_sync.call_sync(location="Seoul") == 35
    assert tool_sync.call_sync(location="Seoul", unit="Fahrenheit") == 95


async def test_mcp_tools():
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    mcp_client = await ai.MCPClient.from_stdio("uvx", ["mcp-server-time"])

    # call tools() to get list of MCP tools
    current_time_tool = mcp_client.tools[0]
    tool_result = json.loads(await current_time_tool(timezone="Asia/Seoul"))
    assert tool_result["timezone"] == "Asia/Seoul"
    assert abs(
        datetime.fromisoformat(tool_result["datetime"])
        - datetime.now(tz=ZoneInfo("Asia/Seoul"))
    ) < timedelta(seconds=5)

    # call get_tool(name) to get the MCP tool with the given name
    current_time_tool = mcp_client.get_tool("get_current_time")
    tool_result = json.loads(current_time_tool.call_sync(timezone="Asia/Seoul"))
    assert tool_result["timezone"] == "Asia/Seoul"
    assert abs(
        datetime.fromisoformat(tool_result["datetime"])
        - datetime.now(tz=ZoneInfo("Asia/Seoul"))
    ) < timedelta(seconds=5)
