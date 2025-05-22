import type { NDArray } from "ailoy_addon.node";

import { Runtime, generateUUID } from "./runtime";

export type EmbeddingModelName = "bge-m3";

export type VectorStoreName = "faiss" | "chromadb";

interface EmbeddingModelDescription {
  modelId: string;
  componentType: "tvm_embedding_model";
  dimension?: number;
}

const modelDescriptions: Record<EmbeddingModelName, EmbeddingModelDescription> =
  {
    "bge-m3": {
      modelId: "BAAI/bge-m3",
      componentType: "tvm_embedding_model",
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
  private runtime: Runtime;

  private componentState: {
    embeddingName: string;
    vecstoreName: string;
    valid: boolean;
  };

  constructor(runtime: Runtime) {
    this.runtime = runtime;

    this.componentState = {
      embeddingName: generateUUID(),
      vecstoreName: generateUUID(),
      valid: false,
    };
  }

  /**
   * Defines the embedding model and vector store components to the runtime.
   * This must be called before using any other method in the class. If already defined, this is a no-op.
   */
  async define(
    embeddingModelName: EmbeddingModelName,
    embeddingModelAttrs: Record<string, any>,
    vectorStoreName: VectorStoreName,
    vectorStoreAttrs: Record<string, any>
  ): Promise<void> {
    // Skip if the component already exists
    if (this.componentState.valid) return;

    if (!(embeddingModelName in modelDescriptions))
      throw Error(`Model "${embeddingModelName}" is not available`);
    const modelDesc = modelDescriptions[embeddingModelName];

    // Add model name into attrs
    if (!embeddingModelAttrs.model)
      embeddingModelAttrs.model = modelDesc.modelId;

    // Initialize embedding model
    const result1 = await this.runtime.define(
      modelDesc.componentType,
      this.componentState.embeddingName,
      embeddingModelAttrs
    );
    if (!result1) throw Error("Embedding model define failed");

    // Add dimension into attrs
    if (!vectorStoreAttrs.dimension && modelDesc.dimension)
      vectorStoreAttrs.dimension = modelDesc.dimension;

    // Initialize vector store
    const vsComponentType = {
      faiss: "faiss_vector_store",
      chromadb: "chromadb_vector_store",
    }[vectorStoreName];
    const result2 = await this.runtime.define(
      vsComponentType,
      this.componentState.vecstoreName,
      vectorStoreAttrs
    );
    if (!result2) throw Error("Embedding model define failed");

    this.componentState.valid = true;
  }

  /**
   * Delete resources from the runtime.
   * This should be called when the VectorStore is no longer needed. If already deleted, this is a no-op.
   */
  async delete(): Promise<void> {
    if (!this.componentState.valid) return;
    await this.runtime.delete(this.componentState.embeddingName);
    await this.runtime.delete(this.componentState.vecstoreName);
    this.componentState.valid = false;
  }

  /** Inserts a new document into the vector store */
  async insert(item: VectorStoreInsertItem): Promise<void> {
    const embedding = await this.embedding(item.document);
    await this.runtime.callMethod(this.componentState.vecstoreName, "insert", {
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
    const embedding = await this.embedding(query);
    const resp: { results: Array<VectorStoreRetrieveItem> } =
      await this.runtime.callMethod(
        this.componentState.vecstoreName,
        "retrieve",
        {
          query_embedding: embedding,
          top_k: topK,
        }
      );
    return resp.results;
  }

  async clear(): Promise<void> {
    await this.runtime.callMethod(this.componentState.vecstoreName, "clear");
  }

  /** Generates an embedding vector for the given input text using the embedding model */
  async embedding(
    /** Input text to embed */
    text: string
  ): Promise<NDArray> {
    const resp: { embedding: NDArray } = await this.runtime.callMethod(
      this.componentState.embeddingName,
      "infer",
      {
        prompt: text,
      }
    );
    return resp.embedding;
  }
}

export async function defineVectorStore(
  runtime: Runtime,
  modelName: EmbeddingModelName,
  vectorStoreName: VectorStoreName,
  args?: {
    device?: number;
    url?: string;
    collection?: string;
  }
): Promise<VectorStore> {
  const vs = new VectorStore(runtime);

  const args_ = args || {};

  // Attribute input for call `rt.define(embedding_model)`
  let embeddingAttrs: Record<string, any> = {};
  if (args_.device) embeddingAttrs["device"] = args_.device;

  // Attribute input for call `rt.define(vector_store)`
  let vectorStoreAttrs: Record<string, any> = {};
  if (args_.url) vectorStoreAttrs["url"] = args_.url;
  if (args_.collection) vectorStoreAttrs["collection"] = args_.collection;

  await vs.define(modelName, embeddingAttrs, vectorStoreName, vectorStoreAttrs);

  return vs;
}
