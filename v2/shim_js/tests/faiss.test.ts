import { describe, it, expect } from "vitest";

import { init_faiss_vector_store } from "../src/index";

describe("Faiss VectorStore", async () => {
  it("VectorStore operations", async () => {
    const vs = await init_faiss_vector_store({
      dimension: 10,
      description: "IDMap2,Flat",
      metric: "METRIC_L2",
    });

    expect(vs.get_metric_type()).to.be.equal("METRIC_L2");
    expect(vs.get_dimension()).to.be.equal(10);

    // add two vectors
    vs.add_vectors_with_ids([
      {
        vector: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        id: 1,
      },
      {
        vector: new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        id: 2,
      },
    ]);
    expect(vs.get_ntotal()).to.be.equal(2);

    // get by id
    const result = vs.get_by_ids([1]);
    expect(result[0]).to.deep.equal(
      new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    );

    // search top 2 vectors
    const searchResults = vs.search_vectors(
      [new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
      2
    );
    expect(searchResults[0].length).to.be.equal(2);

    // id of 0 should be 1
    expect(searchResults[0][0].id).to.be.equal(1);
    // id of 1 should be 2
    expect(searchResults[0][1].id).to.be.equal(2);

    // distance of 0 should be less than 1
    expect(searchResults[0][0].distance).to.be.lessThan(
      searchResults[0][1].distance
    );

    // clear
    vs.clear();
    expect(vs.get_ntotal()).to.be.equal(0);
  });
});
