import io
import os
from urllib.request import urlopen

import pytest
from PIL import Image

import ailoy as ai

pytestmark = [
    pytest.mark.agent,
    pytest.mark.agent_openai,
    pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def runtime():
    with ai.Runtime() as rt:
        yield rt


@pytest.fixture(scope="module")
def _agent(runtime: ai.Runtime):
    with ai.Agent(runtime, ai.OpenAIModel(id="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))) as agent:
        yield agent


@pytest.fixture(scope="function")
def agent(_agent: ai.Agent):
    _agent.clear_messages()
    _agent.clear_tools()
    return _agent


test_image_url = "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385"
test_audio_url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"


def test_tool_call_calculator(agent: ai.Agent):
    agent.add_tools_from_preset("calculator")

    query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))"
    print(f"\nQuery: {query}")
    for resp in agent.query(query):
        agent.print(resp)


def test_image_input_from_url(agent: ai.Agent):
    for resp in agent.query(
        [
            "What is in this image",
            ai.AgentInputImageUrl(url=test_image_url),
        ]
    ):
        resp.print()


def test_image_input_from_pillow(agent: ai.Agent):
    with urlopen(test_image_url) as resp:
        image_data = resp.read()
        image = Image.open(io.BytesIO(image_data))

    for resp in agent.query(
        [
            "What is in this image",
            ai.AgentInputImagePillow(image=image),
        ]
    ):
        resp.print()


def test_audio_input_from_base64(runtime: ai.Runtime):
    with ai.Agent(
        runtime,
        ai.OpenAIModel(id="gpt-4o-audio-preview", api_key=os.getenv("OPENAI_API_KEY")),
    ) as agent:
        with urlopen(test_audio_url) as resp:
            audio_data = resp.read()

        for resp in agent.query(
            [
                "What's in these recording?",
                ai.AgentInputAudioBytes(data=audio_data, format="wav"),
            ]
        ):
            agent.print(resp)
