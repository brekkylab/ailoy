import time

import pytest

from ailoy.runtime import AsyncRuntime

pytestmark = [pytest.mark.runtime, pytest.mark.asyncio]


@pytest.fixture(scope="module")
def runtime():
    print("creating async runtime")
    time.sleep(3)
    with AsyncRuntime("inproc://async") as rt:
        yield rt


async def test_async_echo(runtime: AsyncRuntime):
    text = "hello world"
    resp = await runtime.call("echo", {"text": text})
    assert resp["text"] == text


async def test_async_spell(runtime: AsyncRuntime):
    text = "abcdefghijk"
    i = 0
    async for out in runtime.call_iter("spell", {"text": text}):
        assert out["text"] == text[i]
        i += 1


async def test_async_infer_language_model(runtime: AsyncRuntime):
    await runtime.define("tvm_language_model", "lm0", {"model": "Qwen/Qwen3-0.6B"})
    input = {"messages": [{"role": "user", "content": "Who are you?"}]}
    print("\n")
    async for out in runtime.call_iter_method("lm0", "infer", input):
        print(out["message"]["content"], end="", flush=True)
    await runtime.delete("lm0")
