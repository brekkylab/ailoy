import asyncio
import os
from typing import Literal

import pytest

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest.fixture(scope="module")
def qwen3():
    return ai.LangModel.new_local_sync("Qwen/Qwen3-4B", progress_callback=print)


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


@pytest.fixture(
    params=[
        "What is your name?",
        [
            ai.Message(
                role="system", contents="You are a helpful assistant with name 'Ailoy'."
            ),
            ai.Message(role="user", contents="What is your name?"),
        ],
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
    ]
)
def simple_chat_messages(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
async def test_simple_chat(agent: ai.Agent, simple_chat_messages):
    finish_reason = None
    acc = ai.MessageDelta()
    async for resp in agent.run_delta(
        simple_chat_messages,
        config=ai.AgentConfig(
            inference=ai.InferenceConfig(temperature=0.0, think_effort="disable")
        ),
    ):
        acc += resp.delta
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            result = acc.finish()
            acc = ai.MessageDelta()
        else:
            for content in resp.delta.contents:
                if isinstance(content, ai.PartDelta.Text):
                    print(content.text, end="")
    print()

    assert finish_reason == ai.FinishReason.Stop()
    print(f"{result.contents[0].text=}")


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
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
            config=ai.AgentConfig(
                inference=ai.InferenceConfig(temperature=0.0, think_effort="disable")
            ),
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

    acc = ai.MessageDelta()
    results = []
    async for resp in agent.run_delta(
        "List the files in the current directory.",
        config=ai.AgentConfig(
            inference=ai.InferenceConfig(temperature=0.0, think_effort="disable")
        ),
    ):
        acc += resp.delta
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            results.append(acc.finish())
            acc = ai.MessageDelta()
        else:
            for content in resp.delta.contents:
                if isinstance(content, ai.PartDelta.Text):
                    print(content.text, end="")
                elif isinstance(content, ai.PartDelta.Value):
                    print(content.value)
                else:
                    raise ValueError(
                        f"Content has invalid part_type: {content.part_type}"
                    )
            for tool_call in resp.delta.tool_calls:
                if isinstance(tool_call, ai.PartDelta.Function):
                    print(tool_call.function.text, end="")
                else:
                    raise ValueError(
                        f"Tool call has invalid part_type: {tool_call.part_type}"
                    )
    print()

    assert finish_reason == ai.FinishReason.Stop()
    if finish_reason == ai.FinishReason.ToolCall():
        print(results)
    assert results[0].tool_calls[0].function.name == "terminal"
    print(f"{results[1].contents[0].value=}")
    print(f"{results[2].contents[0].text=}")


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

    acc = ai.MessageDelta()
    results = []
    async for resp in agent.run_delta(
        "What is the temperature in Seoul now?",
        config=ai.AgentConfig(
            inference=ai.InferenceConfig(temperature=0.0, think_effort="disable")
        ),
    ):
        acc += resp.delta
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            results.append(acc.finish())
            acc = ai.MessageDelta()
        else:
            for content in resp.delta.contents:
                if isinstance(content, ai.PartDelta.Text):
                    print(content.text, end="")
                elif isinstance(content, ai.PartDelta.Value):
                    print(content.value)
                else:
                    raise ValueError(
                        f"Content has invalid part_type: {content.part_type}"
                    )
            for tool_call in resp.delta.tool_calls:
                if isinstance(tool_call, ai.PartDelta.Function):
                    print(tool_call.function.text, end="")
                else:
                    raise ValueError(
                        f"Tool call has invalid part_type: {tool_call.part_type}"
                    )
    print()

    assert finish_reason == ai.FinishReason.Stop()
    assert results[0].tool_calls[0].function.name == "tool_temperature"
    print(f"{results[1].contents[0].value=}")
    print(f"{results[2].contents[0].text=}")

    agent.remove_tool(tool.get_description().name)


# @pytest.mark.parametrize(
#     "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
# )
# async def test_parallel_tool_call(agent: ai.Agent):
#     async def tool_temperature(
#         location: str, unit: Literal["Celsius", "Fahrenheit"] = "Celsius"
#     ):
#         """
#         Get temperature of the provided location
#         Args:
#             location: The city name
#             unit: The unit of temperature
#         Returns:
#             int: The temperature
#         """
#         await asyncio.sleep(1.0)
#         return 35 if unit == "Celsius" else 95

#     async def tool_wind_speed(location: str):
#         """
#         Get the current wind speed in km/h at a given location
#         Args:
#             location: The city name
#         Returns:
#             float: The current wind speed at the given location in km/h, as a float.
#         """
#         await asyncio.sleep(1.0)
#         return 23.5

#     tools = [
#         ai.Tool.new_py_function(tool_temperature),
#         ai.Tool.new_py_function(tool_wind_speed),
#     ]
#     agent.add_tools(tools)

#     async for resp in agent.run(
#         "Tell me the weather in Seoul both temperature and wind.",
#         config=ai.AgentConfig(inference=ai.InferenceConfig(think_effort="disable")),
#     ):
#         for content in resp.message.contents:
#             if isinstance(content, ai.Part.Text):
#                 print(f"{content.text=}")
#             elif isinstance(content, ai.Part.Value):
#                 print(f"{content.value=}")
#             else:
#                 raise ValueError(f"Content has invalid part_type: {content.part_type}")
#         for tool_call in resp.message.tool_calls:
#             if isinstance(tool_call, ai.Part.Function):
#                 print(
#                     f"function_call={tool_call.function.name}(**{tool_call.function.arguments})"
#                 )
#             else:
#                 raise ValueError(
#                     f"Tool call has invalid part_type: {tool_call.part_type}"
#                 )

#     agent.remove_tools([t.get_description().name for t in tools])


@pytest.mark.parametrize(
    "agent", ["qwen3", "openai", "gemini", "claude", "grok"], indirect=True
)
async def test_mcp_tools(agent: ai.Agent):
    mcp_client = await ai.MCPClient.from_stdio("uvx", ["mcp-server-time"])
    tools = mcp_client.tools
    agent.add_tools(tools)

    acc = ai.MessageDelta()
    results = []
    async for resp in agent.run_delta(
        "What time is it currently in Asia/Seoul?",
        config=ai.AgentConfig(
            inference=ai.InferenceConfig(temperature=0.0, think_effort="disable")
        ),
    ):
        acc += resp.delta
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
            results.append(acc.finish())
            acc = ai.MessageDelta()
        else:
            for content in resp.delta.contents:
                if isinstance(content, ai.PartDelta.Text):
                    print(content.text, end="")
                elif isinstance(content, ai.PartDelta.Value):
                    print(content.value)
                else:
                    raise ValueError(
                        f"Content has invalid part_type: {content.part_type}"
                    )
            for tool_call in resp.delta.tool_calls:
                if isinstance(tool_call, ai.PartDelta.Function):
                    print(tool_call.function.text, end="")
                else:
                    raise ValueError(
                        f"Tool call has invalid part_type: {tool_call.part_type}"
                    )
    print()

    assert finish_reason == ai.FinishReason.Stop()
    if results[0].tool_calls[0].function.name != "get_current_time":
        print(results)

    assert results[0].tool_calls[0].function.name == "get_current_time"
    print(f"{results[1].contents[0].value=}")
    print(f"{results[2].contents[0].text=}")

    agent.remove_tools([t.get_description().name for t in tools])


@pytest.mark.parametrize("agent", ["qwen3"], indirect=True)
async def test_knowledge_with_polyfill(agent: ai.Agent):
    vs = ai.VectorStore.new_faiss(1024)
    emb = await ai.EmbeddingModel.new_local("BAAI/bge-m3")

    doc0 = "Ailoy is an awesome AI agent framework supporting Rust, Python, Nodejs and WebAssembly."
    emb0 = await emb.infer(doc0)
    vs.add_vector(ai.VectorStoreAddInput(embedding=emb0, document=doc0))

    knowledge = ai.Knowledge.new_vector_store(vs, emb)
    agent.set_knowledge(knowledge)

    acc = ai.MessageDelta()
    async for resp in agent.run_delta(
        "What is Ailoy?",
        config=ai.AgentConfig.from_dict({"inference": {"document_polyfill": "Qwen3"}}),
    ):
        acc += resp.delta
        if resp.finish_reason is not None:
            acc = ai.MessageDelta()
        else:
            for content in resp.delta.contents:
                if isinstance(content, ai.PartDelta.Text):
                    print(content.text, end="")
    print()

    agent.remove_knowledge()
