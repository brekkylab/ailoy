import { describe, expect, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy EmbeddingModel", async () => {
  it("Local(bge-m3)", async () => {
    const model = await ailoy.EmbeddingModel.newLocal("BAAI/bge-m3", {
      progressCallback: (prog) => console.log(prog),
    });
    const emb = await model.infer("What is BGE-M3?");
    expect(emb.length).to.be.equal(1024);
  });
});
