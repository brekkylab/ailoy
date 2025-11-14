import { describe, expect, it } from "vitest";

import * as ailoy from "../src/";

describe("Ailoy Knowledge", async () => {
  it.sequential("VectorStoreKnowledge", async () => {
    const vs = await ailoy.VectorStore.newFaiss(1024);
    const emb = await ailoy.EmbeddingModel.newLocal("BAAI/bge-m3", {
      progressCallback: (prog) => console.log(prog),
    });
    const doc0 =
      "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.";
    const emb0 = await emb.infer(doc0);
    const knowledge = ailoy.Knowledge.newVectorStore(vs, emb);

    await vs.addVector({
      embedding: emb0,
      document: doc0,
    });

    const doc1 =
      "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document";
    const emb1 = await emb.infer(doc1);
    await vs.addVector({
      embedding: emb1,
      document: doc1,
    });

    const result = await knowledge.retrieve("What is BGE-M3?", { topK: 2 });
    console.log(result);
  });
});
