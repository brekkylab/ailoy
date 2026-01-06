import pytest
import pytest_asyncio

import ailoy as ai

pytestmark = [pytest.mark.asyncio]


@pytest_asyncio.fixture(scope="module")
async def model():
    model = await ai.LangModel.new_local(
        "Qwen/Qwen3-0.6B", progress_callback=lambda prog: print(prog)
    )
    return model


async def test_simple_chat_delta(model: ai.LangModel):
    msg_d = ai.MessageDelta()
    async for m in model.infer_delta("Hello"):
        if m.delta.thinking:
            print("thinking: ", m.delta.thinking)
        print(m.delta.contents)
        msg_d += m.delta
    print(msg_d)
    print(msg_d.finish())


async def test_simple_chat(model: ai.LangModel):
    msg = ai.Message("user", contents="Hello")
    msg_r = await model.infer([msg])
    print(msg_r)


async def test_chat_with_think(model: ai.LangModel):
    msg = ai.Message("user", contents=[ai.Part.Text(text="Hello")])
    msg_d = ai.MessageDelta()
    config = ai.LangModelInferConfig(think_effort="enable")
    async for m in model.infer_delta([msg], config=config):
        if m.delta.thinking:
            print("thinking: ", m.delta.thinking)
        print(m.delta.contents)
        msg_d += m.delta
    print(msg_d)
    print(msg_d.finish())
