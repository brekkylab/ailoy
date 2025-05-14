import pytest

from ailoy import AsyncRuntime, Runtime
from ailoy.agent import Agent
from ailoy.vector_store import FAISSConfig, VectorStore

from .common import print_agent_response


def test_echo():
    rt = Runtime("inproc://test_echo")
    rt.call("echo", "hello world") == "hello world"


def test_spell():
    rt = Runtime("inproc://test_spell")
    for i, out in enumerate(rt.call_iter("spell", "abcdefghijk")):
        assert out == "abcdefghijk"[i]


@pytest.mark.asyncio
async def test_async_echo():
    rt = AsyncRuntime("inproc://test_async_echo")
    await rt.call("echo", "hello world") == "hello world"


@pytest.mark.asyncio
async def test_async_spell():
    rt = AsyncRuntime("inproc://test_async_spell")
    i = 0
    async for out in rt.call_iter("spell", "abcdefghijk"):
        assert out == "abcdefghijk"[i]
        i += 1


@pytest.mark.asyncio
async def test_vectorstore():
    rt = AsyncRuntime("inproc://test_vectorstore")
    vs = VectorStore(rt, FAISSConfig())
    await vs.initialize()

    doc1 = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
    doc2 = "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"  # noqa: E501
    await vs.insert(document=doc1, metadata={"value": "BGE-M3"})
    await vs.insert(document=doc2, metadata={"value": "BM25"})

    resp = await vs.retrieve(query="What is BGE M3?", top_k=1)
    assert len(resp) == 1
    assert resp[0].metadata["value"] == "BGE-M3"

    await vs.deinitialize()


@pytest.mark.asyncio
async def test_async_infer_language_model():
    rt = AsyncRuntime("inproc://test_async_infer_language_model")
    await rt.define("tvm_language_model", "lm0", {"model": "Qwen/Qwen3-0.6B"})
    input = {"messages": [{"role": "user", "content": "Who are you?"}]}
    print("\n")
    async for out in rt.call_iter_method("lm0", "infer", input):
        print(out["message"]["content"], end="", flush=True)
    await rt.delete("lm0")


@pytest.mark.asyncio
async def test_tool_call_calculator():
    rt = AsyncRuntime("inproc://test_tool_call_calculator")
    ex = Agent(rt, model_name="qwen3-8b")
    await ex.initialize()

    ex.add_tools_from_preset("calculator")

    query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))"
    async for resp in ex.run(query):
        print_agent_response(resp)

    await ex.deinitialize()
    rt.close()


@pytest.mark.asyncio
async def test_tool_call_frankfurter():
    rt = AsyncRuntime("inproc://test_tool_call_frankfurter")
    ex = Agent(rt, model_name="qwen3-8b")
    await ex.initialize()

    ex.add_tools_from_preset("frankfurter")

    query = "I want to buy 250 U.S. Dollar and 350 Chinese Yuan with my Korean Won. How much do I need to take?"
    async for resp in ex.run(query):
        print_agent_response(resp)

    await ex.deinitialize()
