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
        await asyncio.sleep(1.0)
        return 36

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
        print(f"{location=}")
        print(f"{unit=}")
        return 36

    desc = ai.ToolDesc(
        "temperature",
        "Get temperature of the provided location",
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name",
                },
                "unit": {
                    "type": "string",
                    "description": "temperature unit",
                    "enum": ["Celsius", "Fahrenheit"],
                },
            },
            "required": ["location", "unit"],
        },
    )

    # with async function
    # tool = ai.PythonAsyncFunctionTool(func=tool_temperature)
    tool = ai.Tool.new_py_function(desc, tool_temperature)

    # __call__() (async)
    print(f'{await tool(location="Seoul")=}')
    print(f'{await tool(location="Seoul", unit="Fahrenheit")=}')

    # call() (async)
    print(f'{await tool.call(location="Seoul")=}')
    print(f'{await tool.call(location="Seoul", unit="Fahrenheit")=}')

    # with sync function
    # tool = ai.PythonAsyncFunctionTool(func=tool_temperature_sync)
    tool_sync = ai.Tool.new_py_function(desc, tool_temperature_sync)

    # __call__() (async)
    print(f'{await tool_sync(location="Seoul")=}')
    print(f'{await tool_sync(location="Seoul", unit="Fahrenheit")=}')

    # call() (async)
    print(f'{await tool_sync.call(location="Seoul")=}')
    print(f'{await tool_sync.call(location="Seoul", unit="Fahrenheit")=}')

    # call_sync() (sync)
    print(f'{tool_sync.call_sync(location="Seoul")=}')
    print(f'{tool_sync.call_sync(location="Seoul", unit="Fahrenheit")=}')


async def test_mcp_tools():
    mcp_client = await ai.MCPClient.from_stdio("uvx", ["mcp-server-time"])

    # call tools() to get list of MCP tools
    current_time_tool = mcp_client.tools()[0]
    print(await current_time_tool(timezone="Asia/Seoul"))

    # call get_tool(name) to get the MCP tool with the given name
    current_time_tool = mcp_client.get_tool("get_current_time")
    print(current_time_tool.call_sync(timezone="Asia/Seoul"))
