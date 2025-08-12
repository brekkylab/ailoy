import { describe, it, expect, afterAll } from "vitest";

import { Runtime } from "../src/runtime";
import { defineVectorStore, VectorStore } from "../src/vector_store";

describe("Vectorstore", async () => {
  const rt = new Runtime();
  await rt.start();

  const testVectorStore = async (vs: VectorStore) => {
    const doc0 =
      "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.";
    const meta0 = { topic: "bge-m3" };
    const item0 = await vs.insert({
      document: doc0,
      metadata: meta0,
    });

    const doc1 =
      "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document";
    const meta1 = { topic: "bm25" };
    const item1 = await vs.insert({
      document: doc1,
      metadata: meta1,
    });

    const results = await vs.retrieve("What is BGE-M3?");
    expect(results).to.be.length(2);

    expect(results[0].id).to.be.equal(item0.id);
    expect(results[0].document).to.be.equal(doc0);
    expect(results[0].metadata).to.deep.equal(meta0);
    expect(results[1].id).to.be.equal(item1.id);
    expect(results[1].document).to.be.equal(doc1);
    expect(results[1].metadata).to.deep.equal(meta1);
  };

  it.sequential("FAISS Vectorstore", async () => {
    const vs = await defineVectorStore(rt, {
      type: "faiss",
      embedding: {
        modelId: "BAAI/bge-m3",
      },
    });
    await testVectorStore(vs);
    await vs.delete();
  });

  it.sequential("Chroma Vectorstore", async ({ skip }) => {
    const chromadbUrl = "http://localhost:8000";

    try {
      const resp = await fetch(`${chromadbUrl}/api/v2/healthcheck`);
      if (resp.status !== 200) {
        throw Error("Chromadb is not healthy");
      }
    } catch (e) {
      skip("Chromadb is not available. Skip this test.");
    }

    const vs = await defineVectorStore(rt, {
      type: "chromadb",
      url: chromadbUrl,
      embedding: {
        modelId: "BAAI/bge-m3",
      },
    });
    await testVectorStore(vs);
    await vs.delete();
  });

  afterAll(async () => {
    await rt.stop();
  });
});
