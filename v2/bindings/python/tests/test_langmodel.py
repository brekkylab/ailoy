import pytest
import pytest_asyncio

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def model():
    model = await ai.LangModel.CreateLocal(
        "Qwen/Qwen3-0.6B", progress_callback=lambda prog: print(prog)
    )
    return model


async def test_simple_chat_delta(model: ai.LangModel):
    msg = ai.Message(ai.Role.User, contents=[ai.Part.Text(text="Hello")])
    msg_d = ai.MessageDelta(ai.Role.Assistant)
    async for m in model.infer_delta([msg]):
        if m.delta.thinking:
            print("thinking: ", m.delta.thinking)
        print(m.delta.contents)
        msg_d += m.delta
    print(msg_d)
    print(msg_d.to_message())

async def test_simple_chat(model: ai.LangModel):
    msg = ai.Message(ai.Role.User, contents=[ai.Part.Text(text="Hello")])
    msg_r = await model.infer([msg])
    print(msg_r)


async def test_chat_with_think(model: ai.LangModel):
    msg = ai.Message(ai.Role.User, contents=[ai.Part.Text(text="Hello")])
    msg_d = ai.MessageDelta(ai.Role.Assistant)
    config = ai.InferenceConfig(think_effort=ai.ThinkEffort.Enable)
    async for m in model.infer_delta([msg], config=config):
        if m.delta.thinking:
            print("thinking: ", m.delta.thinking)
        print(m.delta.contents)
        msg_d += m.delta
    print(msg_d)
    print(msg_d.to_message())
