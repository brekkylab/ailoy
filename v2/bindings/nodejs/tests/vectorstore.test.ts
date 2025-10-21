import { afterAll, assert, beforeEach, describe, expect, test } from "vitest";

import * as ailoy from "../src";

const configs = [
  {
    name: "Faiss",
    create: async () => {
      return await ailoy.VectorStore.newFaiss(1024);
    },
  },
  {
    name: "Chroma",
    create: async () => {
      const url = "http://localhost:8000";
      const collectionName = "my_collection";
      return await ailoy.VectorStore.newChroma(url, collectionName);
    },
  },
];

const model = await ailoy.EmbeddingModel.newLocal("BAAI/bge-m3");

for (const cfg of configs) {
  let skip = false;
  let vs: ailoy.VectorStore;
  try {
    vs = await cfg.create();
  } catch (err) {
    skip = true;
  }

  describe.skipIf(skip)(`VectorStore: ${cfg.name}`, () => {
    beforeEach(async () => {
      await vs.clear();
    });

    afterAll(async () => {
      await vs.clear();
    });

    test.sequential("Add & get & remove a single input", async () => {
      const document = "What is your name?";
      const embedding = await model.infer(document);
      const input: ailoy.VectorStoreAddInput = {
        embedding,
        document,
      };
      const id = await vs.addVector(input);

      const getResult = await vs.getById(id);
      assert.isNotNull(getResult);
      expect(getResult.id).to.be.equal(id);
      expect(getResult.document).to.be.equal(document);
      expect(getResult.embedding).to.be.deep.equal(embedding);
      assert.isUndefined(getResult.metadata);

      let count = await vs.count();
      expect(count).to.be.equal(1);

      await vs.removeVector(id);
      count = await vs.count();
      expect(count).to.be.equal(0);
    });

    test.sequential("Add & get & remove multiple inputs", async () => {
      const doc0 = "document0";
      const emb0 = await model.infer(doc0);

      const doc1 = "document1";
      const emb1 = await model.infer(doc1);

      const ids = await vs.addVectors([
        {
          embedding: emb0,
          document: doc0,
        },
        {
          embedding: emb1,
          document: doc1,
        },
      ]);
      expect(ids).to.be.length(2);

      const getResults = await vs.getByIds(ids);
      expect(getResults).to.be.length(2);
      expect(getResults[0].id).to.be.equal(ids[0]);
      expect(getResults[0].document).to.be.equal(doc0);
      expect(getResults[0].embedding).to.be.deep.equal(emb0);
      expect(getResults[1].id).to.be.equal(ids[1]);
      expect(getResults[1].document).to.be.equal(doc1);
      expect(getResults[1].embedding).to.be.deep.equal(emb1);

      let count = await vs.count();
      expect(count).to.be.equal(2);

      await vs.removeVectors(ids);
      count = await vs.count();
      expect(count).to.be.equal(0);
    });

    test.sequential("Retrieve results for a single query", async () => {
      const doc0 = "Ailoy is an awesome library";
      const emb0 = await model.infer(doc0);

      const doc1 = "Langchain is a library";
      const emb1 = await model.infer(doc1);

      const ids = await vs.addVectors([
        {
          embedding: emb0,
          document: doc0,
        },
        {
          embedding: emb1,
          document: doc1,
        },
      ]);

      const query = "What is Ailoy?";
      const queryEmb = await model.infer(query);
      const retrieveResults = await vs.retrieve(queryEmb, 2);
      expect(retrieveResults).to.be.length(2);
      // The query is about Ailoy, so the most similar item should match to the first input.
      expect(retrieveResults[0].id).to.be.equal(ids[0]);
      expect(retrieveResults[1].id).to.be.equal(ids[1]);
      expect(retrieveResults[0].distance).to.be.lessThan(
        retrieveResults[1].distance
      );
    });

    test.sequential("Retrieve batch results for multiple queries", async () => {
      const doc0 =
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.";
      const emb0 = await model.infer(doc0);

      const doc1 =
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document";
      const emb1 = await model.infer(doc1);

      const ids = await vs.addVectors([
        {
          embedding: emb0,
          document: doc0,
        },
        {
          embedding: emb1,
          document: doc1,
        },
      ]);

      const query0 = "What is BGE-M3?";
      const queryEmb0 = await model.infer(query0);
      const query1 = "Defination of BM25";
      const queryEmb1 = await model.infer(query1);
      const retrieveResults = await vs.batchRetrieve([queryEmb0, queryEmb1], 2);
      expect(retrieveResults).to.be.length(2);

      expect(retrieveResults[0]).to.be.length(2);
      // First query is about BGE-M3, so the most similar item should match to the first input.
      expect(retrieveResults[0][0].id).to.be.equal(ids[0]);
      expect(retrieveResults[0][0].distance).to.be.lessThan(
        retrieveResults[0][1].distance
      );

      expect(retrieveResults[1]).to.be.length(2);
      // Second query is about BM25, so the most similar item should match to the second input.
      expect(retrieveResults[1][0].id).to.be.equal(ids[1]);
      expect(retrieveResults[1][0].distance).to.be.lessThan(
        retrieveResults[1][1].distance
      );
    });
  });
}
