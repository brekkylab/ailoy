import { NDArray } from "./ailoy_js_web";
import { Engine } from "./llm/engine";
import { Tokenizer } from "./llm/tokenizer";
import { Runtime } from "./runtime";

export type EmbeddingModelId = "BAAI/bge-m3";
export type EmbeddingModelQuantization = "q4f16_1";
export type VectorStoreType = "faiss" | "chromadb";

interface EmbeddingModelDescription {
  dimension: number;
}

const modelDescriptions: Record<EmbeddingModelId, EmbeddingModelDescription> = {
  "BAAI/bge-m3": {
    dimension: 1024,
  },
};

export interface VectorStoreInsertItem {
  /** The raw text document to insert */
  document: string;
  /** Metadata records additional information about the document */
  metadata?: Record<string, any>;
}

export interface VectorStoreRetrieveItem extends VectorStoreInsertItem {
  id: string;
  similarity: number;
}

/**
 * The `VectorStore` class provides a high-level abstraction for storing and retrieving documents.
 * It mainly consists of two modules - embedding model and vector store.
 *
 * It supports embedding text using AI and interfacing with pluggable vector store backends such as FAISS or ChromaDB.
 * This class handles initialization, insertion, similarity-based retrieval, and cleanup.
 *
 * Typical usage involves:
 *   1. Initializing the store via `initialize()`
 *   2. Inserting documents via `insert()`
 *   3. Querying similar documents via `retrieve()`
 *
 * The embedding model and vector store are defined dynamically within the provided runtime.
 */
export class VectorStore {
  private runtime: Runtime | undefined = undefined;
  private embeddingModel: Engine | undefined = undefined;
  private vectorstoreComponentName: string | undefined = undefined;
  #initialized: boolean = false;

  /**
   * Defines the embedding model and vector store components to the runtime.
   * This must be called before using any other method in the class. If already defined, this is a no-op.
   */
  async define(
    runtime: Runtime,
    args: {
      embedding?: {
        modelId: EmbeddingModelId;
        quantization: EmbeddingModelQuantization;
      };
      vectorstore?: {
        type: VectorStoreType;
        attrs?: Record<string, any>;
      };
    }
  ): Promise<void> {
    // Skip if the vector store is already initialized
    if (this.#initialized) return;

    this.runtime = runtime;

    /**
     * Preparing Embedding Model
     */
    const embeddingModelId = args.embedding?.modelId ?? "BAAI/bge-m3";
    const embeddingModelQuantization =
      args.embedding?.quantization ?? "q4f16_1";
    const modelDesc = modelDescriptions[embeddingModelId];

    const { results }: { results: Array<{ model_id: string }> } =
      await runtime.call("list_local_models");

    if (!results.some(({ model_id }) => model_id === embeddingModelId)) {
      await runtime.call("download_model", {
        model_id: embeddingModelId,
        quantization: embeddingModelQuantization,
        device: "webgpu",
      });
    }

    const tokenizer = new Tokenizer(
      runtime,
      embeddingModelId,
      embeddingModelQuantization
    );
    await tokenizer.init();

    this.embeddingModel = new Engine(embeddingModelId, tokenizer);
    await this.embeddingModel.loadModel();

    /**
     * Preparing VectorStore
     */
    const vsType = args.vectorstore?.type ?? "faiss";
    const vsAttrs = args.vectorstore?.attrs ?? {};
    if (vsType === "faiss") {
      vsAttrs.dimension = modelDesc.dimension;
    }

    this.vectorstoreComponentName = runtime.generateUUID();
    const vsResult = await this.runtime.define(
      `${vsType}_vector_store`,
      this.vectorstoreComponentName,
      vsAttrs
    );
    if (!vsResult) throw Error("Failed to define VectorStore");

    this.#initialized = true;
  }

  /**
   * Delete resources from the runtime.
   * This should be called when the VectorStore is no longer needed. If already deleted, this is a no-op.
   */
  async delete(): Promise<void> {
    if (!this.#initialized) return;

    await this.embeddingModel!.dispose();
    this.embeddingModel = undefined;

    await this.runtime!.delete(this.vectorstoreComponentName!);
    this.vectorstoreComponentName = undefined;

    this.runtime = undefined;

    this.#initialized = false;
  }

  /** Inserts a new document into the vector store */
  async insert(item: VectorStoreInsertItem): Promise<void> {
    if (!this.#initialized) throw Error("VectorStore not initialized yet");

    const embedding = await this.embedding(item.document);
    await this.runtime!.callMethod(this.vectorstoreComponentName!, "insert", {
      embedding: embedding,
      document: item.document,
      metadata: item.metadata,
    });
  }

  /** Retrieves the top-K most similar documents to the given query */
  async retrieve(
    /** The input query string to search for similar content */
    query: string,
    /** Number of top similar documents to retrieve */
    topK: number = 5
  ): Promise<Array<VectorStoreRetrieveItem>> {
    if (!this.#initialized) throw Error("VectorStore not initialized yet");

    const embedding = await this.embedding(query);
    const resp: { results: Array<VectorStoreRetrieveItem> } =
      await this.runtime!.callMethod(
        this.vectorstoreComponentName!,
        "retrieve",
        {
          query_embedding: embedding,
          top_k: topK,
        }
      );
    return resp.results;
  }

  async clear(): Promise<void> {
    if (!this.#initialized) throw Error("VectorStore not initialized yet");

    await this.runtime!.callMethod(this.vectorstoreComponentName!, "clear");
  }

  /** Generates an embedding vector for the given input text using the embedding model */
  async embedding(
    /** Input text to embed */
    text: string
  ): Promise<Float32Array> {
    if (!this.#initialized) throw Error("Embedding model not initialized yet");

    const { data } = await this.embeddingModel!.inferEM(text);
    // Assume a single batch
    const vector = new Float32Array(data[0].embedding);
    return vector;
  }
}

export async function defineVectorStore(
  runtime: Runtime,
  args: {
    embedding?: {
      modelId: EmbeddingModelId;
      quantization: EmbeddingModelQuantization;
    };
    vectorstore?: {
      type: VectorStoreType;
      attrs?: Record<string, any>;
    };
  }
): Promise<VectorStore> {
  const vs = new VectorStore();
  await vs.define(runtime, args);
  return vs;
}
