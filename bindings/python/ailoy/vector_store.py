from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, TypeAdapter

from ailoy.runtime import Runtime, generate_uuid


class VectorStoreInsertItem(BaseModel):
    document: str
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreRetrieveItem(VectorStoreInsertItem):
    id: str
    similarity: float


class ComponentState(BaseModel):
    embedding_model_name: str
    vector_store_name: str
    valid: bool


class VectorStore:
    """
    The `VectorStore` class provides a high-level abstraction for storing and retrieving documents.
    It mainly consists of two modules - embedding model and vector store.

    It supports embedding text using AI and interfacing with pluggable vector store backends such as FAISS or ChromaDB.
    This class handles initialization, insertion, similarity-based retrieval, and cleanup.

    Typical usage involves:
    1. Initializing the store via `initialize()`
    2. Inserting documents via `insert()`
    3. Querying similar documents via `retrieve()`

    The embedding model and vector store are defined dynamically within the provided runtime.
    """

    def __init__(
        self,
        runtime: Runtime,
        embedding_model_name: Literal["bge-m3"],
        vector_store_name: Literal["faiss", "chromadb"],
        url: Optional[str] = None,
        collection: Optional[str] = None,
        embedding_model_attrs: dict[str, Any] = dict(),
        vector_store_attrs: dict[str, Any] = dict(),
    ):
        """
        Creates an instance.

        :param runtime: The runtime environment used to manage components.
        :param embedding_model_name: The name of the embedding model to use. Currently it only supports `bge-m3`.
        :param url: (ChromaDB only) URL of the database.
        :param collection: (ChromaDB only) The collection name of the database.
        :param vector_store_name: The name of the vector store provider (One of faiss or chromaDB).
        :param embedding_model_attrs: Additional initialization parameters (for `define_component` runtime call).
        :param vector_store_attrs: Additional initialization parameters (for `define_component` runtime call).
        """
        self._runtime = runtime
        self._component_state = ComponentState(
            embedding_model_name=generate_uuid(),
            vector_store_name=generate_uuid(),
            valid=False,
        )
        self.define(
            embedding_model_name,
            vector_store_name,
            url,
            collection,
            embedding_model_attrs,
            vector_store_attrs,
        )

    def __del__(self):
        self.delete()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.delete()

    def define(
        self,
        embedding_model_name: Literal["bge-m3"],
        vector_store_name: Literal["faiss", "chromadb"],
        url: Optional[str] = None,
        collection: Optional[str] = None,
        embedding_model_attrs: dict[str, Any] = dict(),
        vector_store_attrs: dict[str, Any] = dict(),
    ):
        """
        Defines the embedding model and vector store components to the runtime.
        This must be called before using any other method in the class. If already defined, this is a no-op.
        """
        if self._component_state.valid:
            return

        # Dimension size may be different based on embedding model
        dimension = 1024

        # Initialize embedding model
        if embedding_model_name == "bge-m3":
            dimension = 1024
            self._runtime.define(
                "tvm_embedding_model",
                self._component_state.embedding_model_name,
                {"model": "BAAI/bge-m3"},
            )
        else:
            raise NotImplementedError()

        # Initialize vector store
        if vector_store_name == "faiss":
            if "dimension" not in embedding_model_attrs:
                embedding_model_attrs["dimension"] = dimension
            self._runtime.define("faiss_vector_store", self._component_state.vector_store_name, embedding_model_attrs)
        else:
            if "url" not in vector_store_attrs:
                vector_store_attrs["url"] = url
            if "collection" not in vector_store_attrs:
                vector_store_attrs["collection"] = collection
            self._runtime.define("chromadb_vector_store", self._component_state.vector_store_name, vector_store_attrs)

        self._component_state.valid = True

    def delete(self):
        """
        Delete resources from the runtime.
        This should be called when the VectorStore is no longer needed. If already deleted, this is a no-op.
        """
        if not self._component_state.valid:
            return

        self._runtime.delete(self._component_state.embedding_model_name)
        self._runtime.delete(self._component_state.vector_store_name)
        self._component_state.valid = False

    # TODO: add NDArray typing
    def embedding(self, text: str) -> Any:
        """
        Generates an embedding vector for the given input text using the embedding model.

        :param text: Input text to embed.
        :returns: The resulting embedding vector.
        """
        resp = self._runtime.call_method(
            self._component_state.embedding_model_name,
            "infer",
            {
                "prompt": text,
            },
        )
        return resp["embedding"]

    def insert(self, document: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Inserts a new document into the vector store.

        :param document: The raw text document to insert.
        :param metadata: Metadata records additional information about the document.
        """
        embedding = self.embedding(document)
        self._runtime.call_method(
            self._component_state.vector_store_name,
            "insert",
            {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            },
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[VectorStoreRetrieveItem]:
        """
        Retrieves the top-K most similar documents to the given query.

        :param query: The input query string to search for similar content.
        :param top_k: Number of top similar documents to retrieve.
        :returns: A list of retrieved items.
        """
        embedding = self.embedding(query)
        resp = self._runtime.call_method(
            self._component_state.vector_store_name,
            "retrieve",
            {
                "query_embedding": embedding,
                "k": top_k,
            },
        )
        results = TypeAdapter(List[VectorStoreRetrieveItem]).validate_python(resp["results"])
        return results

    def clean(self):
        """
        Removes all entries from the vector store.
        """
        self._runtime.call_method(self._component_state.vector_store_name, "clean", {})
