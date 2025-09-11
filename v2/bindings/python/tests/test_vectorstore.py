import pytest
from pytest import FixtureRequest

import ailoy as ai


@pytest.fixture(scope="module")
def faiss():
    return ai.FaissVectorStore(dim=10)


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
def test_vectorstore_operations(vs: ai.BaseVectorStore):
    vs.clear()

    input0 = ai.VectorStoreAddInput(
        embedding=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        document="Ailoy is an awesome library",
        metadata={"topic": "Ailoy"},
    )
    input1 = ai.VectorStoreAddInput(
        embedding=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        document="Langchain is a library",
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

    # assume the query is more related to Ailoy
    query_embedding = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]
    retrieve_results = vs.retrieve(query_embedding, top_k=2)
    assert len(retrieve_results) == 2
    assert retrieve_results[0].id == id0
    assert retrieve_results[0].distance < retrieve_results[1].distance

    vs.remove_vector(id1)
    assert vs.count() == 1

    vs.clear()
    assert vs.count() == 0
