import { describe, it, expect } from "vitest";

import { init_faiss_index_wrapper } from "../src/index";

describe("FaissIndexWrapper", async () => {
  it("Basic Operations", async () => {
    const vs = await init_faiss_index_wrapper({
      dimension: 10,
      description: "IDMap2,Flat",
      metric: "L2",
    });

    expect(vs.get_metric_type()).to.be.equal("L2");
    expect(vs.get_dimension()).to.be.equal(10);

    // add two vectors
    vs.add_vectors_with_ids(
      new Float32Array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
      ]),
      2,
      new BigInt64Array([0n, 1n])
    );
    expect(vs.get_ntotal()).to.be.equal(2);

    // get by id
    const result0 = vs.get_by_ids(new BigInt64Array([0n]));
    expect(result0).to.deep.equal(
      new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    );
    const result1 = vs.get_by_ids(new BigInt64Array([1n]));
    expect(result1).to.deep.equal(
      new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    );

    // search top 2 vectors
    const searchResults = vs.search_vectors(
      new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      2
    );

    // id of 0 should be 0
    expect(searchResults.indexes[0]).to.be.equal(0n);
    // id of 1 should be 1
    expect(searchResults.indexes[1]).to.be.equal(1n);

    // distance of 0 should be less than 1
    expect(searchResults.distances[0]).to.be.lessThan(
      searchResults.distances[1]
    );

    // clear
    vs.clear();
    expect(vs.get_ntotal()).to.be.equal(0);
  });
});
