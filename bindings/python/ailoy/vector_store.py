from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, TypeAdapter

from ailoy.runtime import Runtime, generate_uuid


class VectorStoreBaseConfig(BaseModel):
    embedding: Literal["bge-m3"] = "bge-m3"


class FAISSConfig(VectorStoreBaseConfig):
    pass


class ChromadbConfig(VectorStoreBaseConfig):
    url: str
    collection: Optional[str] = None


VectorStoreConfig = Union[FAISSConfig, ChromadbConfig]


class VectorStoreInsertItem(BaseModel):
    document: str
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreRetrieveItem(VectorStoreInsertItem):
    id: str
    similarity: float


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
        config: VectorStoreConfig = FAISSConfig(),
    ):
        """
        Creates an instance.

        :param runtime: The runtime environment used to manage components.
        :param config: Configuration for the vector store, either FAISSConfig or ChromadbConfig.
        """
        self._runtime = runtime
        self._config = config
        self._vector_store_component_name = generate_uuid()
        self._embedding_model_component_name = generate_uuid()
        self._initialized = False

    def __del__(self):
        self.deinitialize()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, traceback):
        self.deinitialize()

    def initialize(self):
        """
        Initializes the embedding model and vector store components.
        This must be called before using any other method in the class. If already initialized, this is a no-op.
        """
        if self._initialized:
            return

        # Dimension size may be different based on embedding model
        dimension = 1024

        # Initialize embedding model
        if self._config.embedding == "bge-m3":
            dimension = 1024
            self._runtime.define("tvm_embedding_model", self._embedding_model_component_name, {"model": "BAAI/bge-m3"})

        # Initialize vector store
        if isinstance(self._config, FAISSConfig):
            self._runtime.define("faiss_vector_store", self._vector_store_component_name, {"dimension": dimension})
        elif isinstance(self._config, ChromadbConfig):
            self._runtime.define(
                "chromadb_vector_store",
                self._vector_store_component_name,
                {
                    "url": self._config.url,
                    "collection": self._config.collection,
                },
            )
        else:
            raise ValueError(f"Unsupported vector store: {type(self._config)}")

        self._initialized = True

    def deinitialize(self):
        """
        Deinitializes all internal components and releases resources.
        This should be called when the VectorStore is no longer needed. If already deinitialized, this is a no-op.
        """
        if not self._initialized:
            return

        self._runtime.delete(self._embedding_model_component_name)
        self._runtime.delete(self._vector_store_component_name)
        self._initialized = False

    # TODO: add NDArray typing
    def embedding(self, text: str) -> Any:
        """
        Generates an embedding vector for the given input text using the embedding model.

        :param text: Input text to embed.
        :returns: The resulting embedding vector.
        """
        resp = self._runtime.call_method(
            self._embedding_model_component_name,
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
            self._vector_store_component_name,
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
            self._vector_store_component_name,
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
        self._runtime.call_method(self._vector_store_component_name, "clean", {})
