import asyncio
from typing import Literal

import pytest
import pytest_asyncio

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def agent():
    model = await ai.LangModel.CreateLocal(
        "Qwen/Qwen3-0.6B", progress_callback=lambda prog: print(prog)
    )
    agent = ai.Agent(model)
    return agent


async def test_simple_chat(agent: ai.Agent):
    async for resp in agent.run(
        [
            ai.Message(
                role=ai.Role.System,
                contents=[
                    ai.Part.Text(text="You are a helpful assistant with name 'Ailoy'.")
                ],
            ),
            ai.Message(
                role=ai.Role.User,
                contents=[ai.Part.Text(text="What is your name?")],
            ),
        ],
        config=ai.InferenceConfig(think_effort=ai.ThinkEffort.Disable),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = resp.aggregated
        else:
            for content in resp.delta.contents:
                if content.part_type == "text":
                    print(content.text, end="")
                elif content.part_type == "function":
                    print(content.function.text, end="")
                elif content.part_type == "value":
                    print(content.value)
                # elif content.part_type == "image":
                #     pass
                else:
                    continue
    print()
    assert finish_reason == ai.FinishReason.Stop()
    print(f"{result.contents[0].text=}")


async def test_simple_qna(agent: ai.Agent):
    qna = [
        ("My favorite color is blue. Please remember that.", ""),
        ("What’s my favorite color?", "blue"),
        ("Actually, my favorite color is gray now. Don't forget this.", ""),
        ("What’s my favorite color?", "gray"),
    ]

    messages = []
    for question, answer in qna:
        messages.append(
            ai.Message(
                role=ai.Role.User,
                contents=[ai.Part.Text(text=question)],
            )
        )
        async for resp in agent.run(
            messages,
            config=ai.InferenceConfig(think_effort=ai.ThinkEffort.Disable),
        ):
            if resp.finish_reason is not None:
                result = resp.aggregated
        messages.append(result)

        full_text = "".join(part.text for part in result.contents)
        assert answer in full_text.lower()


async def test_builtin_tool(agent: ai.Agent):
    tool = ai.Tool.terminal()
    agent.add_tool(tool)
    async for resp in agent.run(
        [
            ai.Message(
                role=ai.Role.User,
                contents=[
                    ai.Part.Text(text="List the files in the current directory.")
                ],
            )
        ],
        config=ai.InferenceConfig(think_effort=ai.ThinkEffort.Disable),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = resp.aggregated
        else:
            for content in resp.delta.contents:
                if content.part_type == "text":
                    print(content.text, end="")
                elif content.part_type == "function":
                    print(content.function.text, end="")
                elif content.part_type == "value":
                    print(content.value)
                # elif content.part_type == "image":
                #     pass
                else:
                    continue
    print()
    assert finish_reason == ai.FinishReason.Stop()
    print(f"{result.contents[0].text=}")


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
        return 35 if unit == "Celsius" else 95

    tool = ai.Tool.new_py_function(tool_temperature)

    agent.add_tool(tool)
    async for resp in agent.run(
        [
            ai.Message(
                role=ai.Role.User,
                contents=[ai.Part.Text(text="What is the temperature in Seoul now?")],
            )
        ],
        config=ai.InferenceConfig(think_effort=ai.ThinkEffort.Disable),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = resp.aggregated
        else:
            for content in resp.delta.contents:
                if content.part_type == "text":
                    print(content.text, end="")
                elif content.part_type == "function":
                    print(content.function.text, end="")
                elif content.part_type == "value":
                    print(content.value)
                # elif content.part_type == "image":
                #     pass
                else:
                    continue
    print()
    assert finish_reason == ai.FinishReason.Stop()
    print(f"{result.contents[0].text=}")

    agent.remove_tool(tool.get_description().name)


async def test_mcp_tools(agent: ai.Agent):
    mcp_client = await ai.MCPClient.from_stdio("uvx", ["mcp-server-time"])
    tools = mcp_client.tools

    agent.add_tools(tools)
    async for resp in agent.run(
        [
            ai.Message(
                role=ai.Role.User,
                contents=[ai.Part.Text(text="What time is it now in Asia/Seoul?")],
            )
        ],
        config=ai.InferenceConfig(think_effort=ai.ThinkEffort.Disable),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = resp.aggregated
        else:
            for content in resp.delta.contents:
                if content.part_type == "text":
                    print(content.text, end="")
                elif content.part_type == "function":
                    print(content.function.text, end="")
                elif content.part_type == "value":
                    print(content.value)
                # elif content.part_type == "image":
                #     pass
                else:
                    continue
    print()
    assert finish_reason == ai.FinishReason.Stop()
    print(f"{result.contents[0].text=}")

    agent.remove_tools([t.get_description().name for t in tools])
