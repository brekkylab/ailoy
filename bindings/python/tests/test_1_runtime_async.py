import time

import pytest

from ailoy.runtime import AsyncRuntime
from ailoy.vector_store import FAISSConfig, VectorStore

pytestmark = [pytest.mark.runtime, pytest.mark.asyncio]


@pytest.fixture(scope="module")
def async_runtime():
    print("creating async runtime")
    time.sleep(3)
    rt = AsyncRuntime("inproc://async")
    yield rt
    rt.close()


async def test_async_echo(async_runtime: AsyncRuntime):
    await async_runtime.call("echo", "hello world") == "hello world"


async def test_async_spell(async_runtime: AsyncRuntime):
    i = 0
    async for out in async_runtime.call_iter("spell", "abcdefghijk"):
        assert out == "abcdefghijk"[i]
        i += 1


async def test_vectorstore(async_runtime: AsyncRuntime):
    vs = VectorStore(async_runtime, FAISSConfig())
    await vs.initialize()

    doc1 = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
    doc2 = "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"  # noqa: E501
    await vs.insert(document=doc1, metadata={"value": "BGE-M3"})
    await vs.insert(document=doc2, metadata={"value": "BM25"})

    resp = await vs.retrieve(query="What is BGE M3?", top_k=1)
    assert len(resp) == 1
    assert resp[0].metadata["value"] == "BGE-M3"

    await vs.deinitialize()


async def test_async_infer_language_model(async_runtime: AsyncRuntime):
    await async_runtime.define("tvm_language_model", "lm0", {"model": "Qwen/Qwen3-0.6B"})
    input = {"messages": [{"role": "user", "content": "Who are you?"}]}
    print("\n")
    async for out in async_runtime.call_iter_method("lm0", "infer", input):
        print(out["message"]["content"], end="", flush=True)
    await async_runtime.delete("lm0")
