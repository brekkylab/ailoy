import base64
import os
from urllib.request import urlopen

import pytest

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


@pytest.fixture(scope="module")
def img_b64_part() -> ai.Part:
    with urlopen(image_url) as resp:
        image_data = resp.read()
        b64_data = base64.b64encode(image_data).decode("utf-8")
        img_part = ai.Part.image_from_base64(b64_data)
        return img_part


@pytest.mark.parametrize("agent", ["openai", "gemini", "claude", "grok"], indirect=True)
async def test_image_base64_input(agent: ai.Agent, img_b64_part: ai.Part):
    async for resp in agent.run(
        [
            ai.Message(
                "user",
                contents=[img_b64_part, ai.Part.Text("What is shown in this image?")],
            )
        ]
    ):
        if resp.aggregated:
            print(resp.aggregated)


@pytest.mark.parametrize("agent", ["openai", "grok"], indirect=True)
async def test_image_url_input(agent: ai.Agent):
    img_part = ai.Part.image_from_url(image_url)
    async for resp in agent.run(
        [
            ai.Message(
                "user",
                contents=[img_part, ai.Part.Text("What is shown in this image?")],
            )
        ]
    ):
        if resp.aggregated:
            print(resp.aggregated)
