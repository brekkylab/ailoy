import type { NDArray } from "ailoy_addon.node";

import { Runtime, generateUUID } from "./runtime";

interface BaseVectorStoreArgs {
  type: string;
  embedding: "bge-m3";
}

interface FAISSArgs extends BaseVectorStoreArgs {
  type: "faiss";
}

interface ChromadbArgs extends BaseVectorStoreArgs {
  type: "chromadb";
  url: string;
  collection?: string;
}

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
  private args: FAISSArgs | ChromadbArgs;
  private vectorstoreComponentName: string;
  private embeddingModelComponentName: string;
  private initialized: boolean;

  constructor(
    /** The runtime environment used to manage components */
    runtime: Runtime,
    /** Configuration for the vector store, either FAISSConfig or ChromadbConfig */
    args: FAISSArgs | ChromadbArgs
  ) {
    this.runtime = runtime;
    this.args = args;
    this.vectorstoreComponentName = generateUUID();
    this.embeddingModelComponentName = generateUUID();
    this.initialized = false;
  }

  /**
   * Initializes the embedding model and vector store components.
   * This must be called before using any other method in the class. If already initialized, this is a no-op.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Dimension size may be different based on embedding model
    let dimension: number = 1024;

    // Initialize embedding model
    if (this.args.embedding === "bge-m3") {
      dimension = 1024;
      await this.runtime.define(
        "tvm_embedding_model",
        this.embeddingModelComponentName,
        {
          model: "BAAI/bge-m3",
        }
      );
    }

    // Initialize vector store
    if (this.args.type === "faiss") {
      await this.runtime.define(
        "faiss_vector_store",
        this.vectorstoreComponentName,
        {
          dimension,
        }
      );
    } else if (this.args.type === "chromadb") {
      await this.runtime.define(
        "chromadb_vector_store",
        this.vectorstoreComponentName,
        {
          url: this.args.url,
          collection: this.args.collection,
        }
      );
    }

    this.initialized = true;
  }

  /**
   * Deinitializes all internal components and releases resources.
   * This should be called when the VectorStore is no longer needed. If already deinitialized, this is a no-op.
   */
  async deinitialize(): Promise<void> {
    if (!this.initialized) return;
    await this.runtime.delete(this.embeddingModelComponentName);
    await this.runtime.delete(this.vectorstoreComponentName);
    this.initialized = false;
  }

  /** Inserts a new document into the vector store */
  async insert(item: VectorStoreInsertItem): Promise<void> {
    const embedding = await this.embedding(item.document);
    await this.runtime.callMethod(this.vectorstoreComponentName, "insert", {
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
      await this.runtime.callMethod(this.vectorstoreComponentName, "retrieve", {
        query_embedding: embedding,
        k: topK,
      });
    return resp.results;
  }

  async clear(): Promise<void> {
    await this.runtime.callMethod(this.vectorstoreComponentName, "clear");
  }

  /** Generates an embedding vector for the given input text using the embedding model */
  async embedding(
    /** Input text to embed */
    text: string
  ): Promise<NDArray> {
    const resp: { embedding: NDArray } = await this.runtime.callMethod(
      this.embeddingModelComponentName,
      "infer",
      {
        prompt: text,
      }
    );
    return resp.embedding;
  }
}
