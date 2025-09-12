import base64
import os
from urllib.request import urlopen

import pytest
from pytest import FixtureRequest

import ailoy as ai

pytestmark = [
    pytest.mark.asyncio,
]

image_url = (
    "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385"
)


@pytest.fixture(scope="module")
def openai():
    if os.getenv("OPENAI_API_KEY") is None:
        pytest.skip("OPEN_API_KEY not set")
    return ai.OpenAILanguageModel(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@pytest.fixture(scope="module")
def gemini():
    if os.getenv("GEMINI_API_KEY") is None:
        pytest.skip("GEMINI_API_KEY not set")
    return ai.GeminiLanguageModel(
        model_name="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")
    )


@pytest.fixture(scope="module")
def anthropic():
    if os.getenv("ANTHROPIC_API_KEY") is None:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ai.AnthropicLanguageModel(
        model_name="claude-sonnet-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY")
    )


@pytest.fixture(scope="module")
def xai():
    if os.getenv("XAI_API_KEY") is None:
        pytest.skip("XAI_API_KEY not set")
    return ai.XAILanguageModel(model_name="grok-4", api_key=os.getenv("XAI_API_KEY"))


@pytest.fixture(scope="module")
def agent(request: FixtureRequest):
    model: ai.BaseLanguageModel = request.getfixturevalue(request.param)
    agent = ai.Agent(model)
    return agent


@pytest.mark.parametrize("agent", ["openai", "xai"], indirect=True)
async def test_image_url_input(agent: ai.Agent):
    agg = ai.MessageAggregator()
    img_part = ai.Part.ImageURL(image_url)
    async for resp in agent.run([img_part, "What is shown in this image?"]):
        msg = agg.update(resp)
        if msg:
            print(msg)


@pytest.mark.parametrize(
    "agent", ["openai", "gemini", "anthropic", "xai"], indirect=True
)
async def test_image_base64_input(agent: ai.Agent):
    with urlopen(image_url) as resp:
        image_data = resp.read()
        b64_data = base64.b64encode(image_data).decode("utf-8")
        img_part = ai.Part.ImageData(b64_data, "image/jpeg")

    agg = ai.MessageAggregator()
    async for resp in agent.run([img_part, "What is shown in this image?"]):
        msg = agg.update(resp)
        if msg:
            print(msg)
