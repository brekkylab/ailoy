import asyncio
import os
from typing import Literal

import pytest

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest.fixture(scope="module")
def qwen3():
    return ai.LangModel.new_local_sync("Qwen/Qwen3-0.6B", progress_callback=print)


@pytest.fixture(scope="module")
def openai():
    if os.getenv("OPENAI_API_KEY") is None:
        pytest.skip("OPEN_API_KEY not set")
    return ai.LangModel.new_stream_api(
        "OpenAI",
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@pytest.fixture(scope="module")
def gemini():
    if os.getenv("GEMINI_API_KEY") is None:
        pytest.skip("GEMINI_API_KEY not set")
    return ai.LangModel.new_stream_api(
        "Gemini", model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")
    )


@pytest.fixture(scope="module")
def claude():
    if os.getenv("ANTHROPIC_API_KEY") is None:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ai.LangModel.new_stream_api(
        "Claude",
        model_name="claude-haiku-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


@pytest.fixture(scope="module")
def grok():
    if os.getenv("XAI_API_KEY") is None:
        pytest.skip("XAI_API_KEY not set")
    return ai.LangModel.new_stream_api(
        "Grok", model_name="grok-4-fast", api_key=os.getenv("XAI_API_KEY")
    )


@pytest.fixture(scope="module")
def agent(request: pytest.FixtureRequest):
    model: ai.LangModel = request.getfixturevalue(request.param)
    agent = ai.Agent(model)
    return agent


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
async def test_simple_chat(agent: ai.Agent):
    acc = ai.MessageDelta()
    async for resp in agent.run_delta(
        [
            ai.Message(
                role="system",
                contents=[
                    ai.Part.Text(text="You are a helpful assistant with name 'Ailoy'.")
                ],
            ),
            ai.Message(
                role="user",
                contents=[ai.Part.Text(text="What is your name?")],
            ),
        ],
        config=ai.InferenceConfig(think_effort="disable"),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = acc.to_message()
        else:
            acc += resp.delta
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


async def test_simple_multiturn(agent: ai.Agent):
    qna = [
        ("John's favorite color is blue. Please remember that.", ""),
        ("What’s John's favorite color?", "blue"),
        ("Actually, John's favorite color is gray now. Don't forget this.", ""),
        ("What’s John's favorite color?", "gray"),
    ]

    messages = []
    for question, answer in qna:
        messages.append(
            ai.Message(
                role="user",
                contents=[ai.Part.Text(text=question)],
            )
        )
        async for resp in agent.run(
            messages,
            config=ai.InferenceConfig(temperature=0.0, think_effort="disable"),
        ):
            result = resp.message
        messages.append(result)

        full_text = "".join(part.text for part in result.contents)
        assert answer in full_text.lower()


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
async def test_builtin_tool(agent: ai.Agent):
    tool = ai.Tool.new_builtin("terminal")
    agent.add_tool(tool)
    acc = ai.MessageDelta("assistant")
    async for resp in agent.run_delta(
        [
            ai.Message(
                role="user",
                contents=[
                    ai.Part.Text(text="List the files in the current directory.")
                ],
            )
        ],
        config=ai.InferenceConfig(think_effort="disable"),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = acc.to_message()
        else:
            acc += resp.delta
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


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
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
    acc = ai.MessageDelta(role="assistant")
    async for resp in agent.run_delta(
        [
            ai.Message(
                role="user",
                contents=[ai.Part.Text(text="What is the temperature in Seoul now?")],
            )
        ],
        config=ai.InferenceConfig(think_effort="disable"),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = acc.to_message()
        else:
            acc += resp.delta
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
    assert finish_reason == ai.FinishReason.Stop()
    print()
    print(f"{result.contents[0].text=}")

    agent.remove_tool(tool.get_description().name)


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
async def test_mcp_tools(agent: ai.Agent):
    mcp_client = await ai.MCPClient.from_stdio("uvx", ["mcp-server-time"])
    tools = mcp_client.tools

    agent.add_tools(tools)
    acc = ai.MessageDelta(role="assistant")
    async for resp in agent.run_delta(
        [
            ai.Message(
                role="user",
                contents=[ai.Part.Text(text="What time is it now in Asia/Seoul?")],
            )
        ],
        config=ai.InferenceConfig(think_effort="disable"),
    ):
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = acc.to_message()
        else:
            acc += resp.delta
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
