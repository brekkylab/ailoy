import pytest
from pytest import FixtureRequest

import ailoy as ai


@pytest.fixture(scope="module")
def emb():
    return ai.LocalEmbeddingModel.create_sync("BAAI/bge-m3")


@pytest.fixture(scope="module")
def faiss():
    return ai.FaissVectorStore(dim=1024)


@pytest.fixture(scope="module")
def chroma():
    url = "http://localhost:8000"
    collection_name = "my_collection"
    try:
        return ai.ChromaVectorStore(chroma_url=url, collection_name=collection_name)
    except RuntimeError:
        pytest.skip(f"ChromaDB is not running on {url}. Skip Chroma test..")


@pytest.fixture
def vs(request: FixtureRequest):
    """Indirect fixture that returns the requested vector store."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("vs", ["faiss", "chroma"], indirect=True)
def test_vectorstore_operations(vs: ai.BaseVectorStore, emb: ai.LocalEmbeddingModel):
    vs.clear()

    doc0 = "Ailoy is an awesome library"
    emb0 = emb.run_sync(doc0)
    input0 = ai.VectorStoreAddInput(
        embedding=emb0,
        document=doc0,
        metadata={"topic": "Ailoy"},
    )

    doc1 = "Langchain is a library"
    emb1 = emb.run_sync(doc1)
    input1 = ai.VectorStoreAddInput(
        embedding=emb1,
        document=doc1,
        metadata={"topic": "Langchain"},
    )

    id0 = vs.add_vector(input0)
    id1 = vs.add_vector(input1)
    assert vs.count() == 2

    get_result0 = vs.get_by_id(id0)
    assert get_result0.id == id0
    assert get_result0.document == input0.document
    assert get_result0.metadata == input0.metadata
    assert get_result0.embedding == input0.embedding

    query = "What is Ailoy?"
    query_emb = emb.run_sync(query)
    retrieve_results = vs.retrieve(query_emb, top_k=2)
    assert len(retrieve_results) == 2
    assert retrieve_results[0].id == id0
    assert retrieve_results[0].distance < retrieve_results[1].distance

    vs.remove_vector(id1)
    assert vs.count() == 1

    vs.clear()
    assert vs.count() == 0
