import pytest

from ailoy import Runtime
from ailoy.vector_store import VectorStore

pytestmark = [pytest.mark.runtime]


@pytest.fixture(scope="module")
def runtime():
    print("creating sync runtime")
    rt = Runtime("inproc://sync")
    yield rt
    rt.stop()


def test_echo(runtime: Runtime):
    text = "hello world"
    resp = runtime.call("echo", {"text": text})
    assert resp["text"] == text


def test_spell(runtime: Runtime):
    text = "abcdefghijk"
    for i, out in enumerate(runtime.call_iter("spell", {"text": text})):
        assert out["text"] == text[i]


def test_vectorstore(runtime: Runtime):
    with VectorStore(runtime, "bge-m3", "faiss") as vs:
        doc1 = "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
        doc2 = "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"  # noqa: E501
        vs.insert(document=doc1, metadata={"value": "BGE-M3"})
        vs.insert(document=doc2, metadata={"value": "BM25"})

        resp = vs.retrieve(query="What is BGE M3?", top_k=1)
        assert len(resp) == 1
        assert resp[0].metadata["value"] == "BGE-M3"
