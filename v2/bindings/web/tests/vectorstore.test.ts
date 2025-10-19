import { describe, expect, it } from "vitest";

import * as ailoy from "../src/";

describe("Ailoy VectorStore", async () => {
  const testVectorStore = async (vs: ailoy.VectorStore) => {
    const doc0 =
      "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.";
    const meta0 = new Map([["topic", "bge-m3"]]);
    const id0 = await vs.addVector({
      embedding: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      document: doc0,
      metadata: meta0,
    });

    const doc1 =
      "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document";
    const meta1 = new Map([["topic", "bm25"]]);
    const id1 = await vs.addVector({
      embedding: new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
      document: doc1,
      metadata: meta1,
    });

    const results = await vs.retrieve(
      new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      2
    );
    expect(results).to.be.length(2);

    expect(results[0].id).to.be.equal(id0);
    expect(results[0].document).to.be.equal(doc0);
    expect(results[0].metadata).to.deep.equal(meta0);
    expect(results[1].id).to.be.equal(id1);
    expect(results[1].document).to.be.equal(doc1);
    expect(results[1].metadata).to.deep.equal(meta1);
  };

  it.sequential("Faiss", async () => {
    const vs = await ailoy.VectorStore.newFaiss(10);
    await testVectorStore(vs);
  });

  it.sequential("Chroma", async ({ skip }) => {
    const chromadbUrl = "http://localhost:8000";

    try {
      const resp = await fetch(`${chromadbUrl}/api/v2/healthcheck`);
      if (resp.status !== 200) {
        throw Error("Chromadb is not healthy");
      }
    } catch (e) {
      skip("Chromadb is not available. Skip this test.");
    }

    const vs = await ailoy.VectorStore.newChroma(chromadbUrl);
    await testVectorStore(vs);
  });
});
