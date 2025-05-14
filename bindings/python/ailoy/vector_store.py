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
    def __init__(
        self,
        runtime: Runtime,
        config: VectorStoreConfig = FAISSConfig(),
    ):
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
        if not self._initialized:
            return

        self._runtime.delete(self._embedding_model_component_name)
        self._runtime.delete(self._vector_store_component_name)
        self._initialized = False

    # TODO: add NDArray typing
    def embedding(self, text: str) -> Any:
        resp = self._runtime.call_method(
            self._embedding_model_component_name,
            "infer",
            {
                "prompt": text,
            },
        )
        return resp["embedding"]

    def insert(self, document: str, metadata: Optional[Dict[str, Any]] = None):
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
        self._runtime.call_method(self._vector_store_component_name, "clean", {})
