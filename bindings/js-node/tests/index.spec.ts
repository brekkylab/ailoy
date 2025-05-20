import { expect } from "chai";

import { Runtime, NDArray } from "../src/index";
import "./agent.spec";

describe("JSVM", () => {
  it("run-echo", async () => {
    const rt = new Runtime("inproc://echo");
    await rt.start();
    const sentence = "hello world";
    const resp = await rt.call("echo", { text: sentence });
    expect(resp.text).to.be.equal(sentence);
    await rt.stop();
  });

  it("run-spell", async () => {
    const rt = new Runtime("inproc://spell");
    await rt.start();
    const sentence = "abcdefghijk";
    let i = 0;
    for await (const resp of rt.callIter("spell", { text: sentence })) {
      expect(resp.text).to.be.equal(sentence[i]);
      i++;
    }
    await rt.stop();
  });

  it("run-multiple", async () => {
    const rt = new Runtime("inproc://multiple");
    await rt.start();
    const sentence = "artificial intelligence";
    const prom1 = rt.callIter("spell", { text: sentence });
    const prom2 = rt.callIter("spell", { text: sentence.toUpperCase() });
    let sentence_reconstruct1 = "";
    let sentence_reconstruct2 = "";
    let prom1_finished = false;
    let prom2_finished = false;
    while (!prom1_finished || !prom2_finished) {
      if (!prom1_finished) {
        const prom1_result = await prom1.next();
        if (prom1_result.done) prom1_finished = true;
        else sentence_reconstruct1 += prom1_result.value.text;
      }
      if (!prom2_finished) {
        const prom2_result = await prom2.next();
        if (prom2_result.done) prom2_finished = true;
        else sentence_reconstruct2 += prom2_result.value.text;
      }
    }
    expect(sentence).to.be.equal(sentence_reconstruct1);
    expect(sentence).to.be.equal(sentence_reconstruct2.toLowerCase());
    await rt.stop();
  });

  it("run-embedding-model", async () => {
    const rt = new Runtime("inproc://embedding_model");
    await rt.start();
    let resp1 = await rt.define("tvm_embedding_model", "em0", {});
    expect(resp1).to.be.equal(true);

    const resp = await rt.callMethod("em0", "infer", {
      prompt: "What is BGE-M3?",
    });
    const arr: NDArray = resp.embedding;
    expect(arr.getShape()).to.deep.equal([1024]);
    await rt.stop();
  });

  it("run-faiss-vector-store", async () => {
    const rt = new Runtime("inproc://faiss_vector_store");
    await rt.start();
    expect(
      await rt.define("faiss_vector_store", "vs0", {
        dimension: 10,
      })
    ).to.be.equal(true);

    const insertInputs = [
      {
        embedding: new Float32Array(
          Array.from({ length: 10 }, () => Math.random())
        ),
        document: "doc1",
        metadata: { value: 1 },
      },
      {
        embedding: new Float32Array(
          Array.from({ length: 10 }, () => Math.random())
        ),
        document: "doc2",
        metadata: null,
      },
    ];

    let vectorIds;
    const resp1 = await rt.callMethod("vs0", "insert_many", insertInputs);
    expect(resp1).to.have.property("ids");
    vectorIds = resp1.ids;

    const resp2 = await rt.callMethod("vs0", "get_by_id", {
      id: vectorIds[0],
    });
    expect(resp2.id).to.be.equal(vectorIds[0]);
    expect(resp2.document).to.be.equal(insertInputs[0].document);
    expect(resp2.metadata).to.deep.equal(insertInputs[0].metadata);

    const resp3 = await rt.callMethod("vs0", "retrieve", {
      query_embedding: insertInputs[0].embedding,
      top_k: 2,
    });
    expect(resp3.results).to.be.lengthOf(2);
    for (const [i, result] of resp3.results.entries()) {
      expect(result.id).to.be.equal(vectorIds[i]);
      expect(result.document).to.be.equal(insertInputs[i].document);
      expect(result.metadata).to.deep.equal(insertInputs[i].metadata);
      // first item should have the highest similarity
      expect(result.similarity).to.be.lessThanOrEqual(
        resp3.results[0].similarity
      );
    }

    const resp4 = await rt.callMethod("vs0", "remove", {
      id: vectorIds[0],
    });
    expect(resp4).to.be.equal(true);

    const resp5 = await rt.callMethod("vs0", "clear", null);
    expect(resp5).to.be.equal(true);

    await rt.delete("vs0");
    await rt.stop();
  });

  it("run-chromadb-vector-store", async () => {
    const rt = new Runtime("inproc://chromadb_vector_store");
    await rt.start();

    let resp = await rt.define("chromadb_vector_store", "vs0", {
      url: "http://localhost:8000",
    });
    expect(resp).to.be.equal(true);

    const insertInputs = [
      {
        embedding: new Float32Array(
          Array.from({ length: 10 }, () => Math.random())
        ),
        document: "doc1",
        metadata: { value: 1 },
      },
      {
        embedding: new Float32Array(
          Array.from({ length: 10 }, () => Math.random())
        ),
        document: "doc2",
        metadata: null,
      },
    ];

    let vectorIds;
    const resp1 = await rt.callMethod("vs0", "insert_many", insertInputs);
    expect(resp1).to.have.property("ids");
    vectorIds = resp1.ids;

    const resp2 = await rt.callMethod("vs0", "get_by_id", {
      id: vectorIds[0],
    });
    expect(resp2.id).to.be.equal(vectorIds[0]);
    expect(resp2.document).to.be.equal(insertInputs[0].document);
    expect(resp2.metadata).to.deep.equal(insertInputs[0].metadata);

    const resp3 = await rt.callMethod("vs0", "retrieve", {
      query_embedding: insertInputs[0].embedding,
      top_k: 2,
    });
    expect(resp3.results).to.be.lengthOf(2);
    for (const [i, result] of resp3.results.entries()) {
      expect(result.id).to.be.equal(vectorIds[i]);
      expect(result.document).to.be.equal(insertInputs[i].document);
      expect(result.metadata).to.deep.equal(insertInputs[i].metadata);
      expect(Number(result.similarity.toFixed(6))).to.be.within(0.0, 1.0);
    }

    const resp4 = await rt.callMethod("vs0", "remove", {
      id: vectorIds[0],
    });
    expect(resp4).to.be.equal(true);

    const resp5 = await rt.callMethod("vs0", "clear", null);
    expect(resp5).to.be.equal(true);

    await rt.delete("vs0");
    await rt.stop();
  });

  it("run-split-text", async () => {
    const rt = new Runtime("inproc://split_text");
    await rt.start();
    const text = `
  Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.

  Last year COVID-19 kept us apart. This year we are finally together again.

  Tonight, we meet as Democrats Republicans and Independents. But most importantly as Americans.
      `;
    const chunkSize = 500;
    const resp = await rt.call("split_text", {
      text,
      chunk_size: chunkSize,
      chunk_overlap: 200,
    });
    for (const chunk of resp.chunks) {
      expect(chunk.length).to.be.lessThan(chunkSize);
    }
    await rt.stop();
  });

  it("run-language-model", async () => {
    const rt = new Runtime("inproc://language_model");
    await rt.start();
    let resp = await rt.define("tvm_language_model", "lm0", {
      model: "meta-llama/Llama-3.2-1B-Instruct",
    });
    expect(resp).to.be.equal(true);
    for await (const resp of rt.callIterMethod("lm0", "infer", {
      messages: [
        { role: "user", content: "Who is the president of US in 2021?" },
      ],
    })) {
      expect(resp).to.have.property("message");
    }
    await rt.stop();
  });

  it("run-http-request", async () => {
    const rt = new Runtime("inproc://http-request");
    await rt.start();
    const args = {
      url: "https://api.frankfurter.dev/v1/latest?base=USD&symbols=KRW",
      method: "GET",
      headers: {},
    };
    const resp = await rt.call("http_request", args);
    expect(resp.status_code).to.be.equal(200);
    expect(resp.headers["Content-Type"])
      .to.be.a("string")
      .and.satisfy((data: string) => data.startsWith("application/json"));

    const body = JSON.parse(resp.body);
    expect(body.amount).to.be.equal(1.0);
    expect(body.base).to.be.equal("USD");
    expect(body.rates).to.have.property("KRW");
    await rt.stop();
  });

  // SKIP OpenAI test if OPENAI_API_KEY is not set
  if (process.env.OPENAI_API_KEY !== undefined) {
    it("run-openai-infer", async () => {
      const rt = new Runtime("inproc://openai");
      await rt.start();
      const resp = await rt.define("openai", "oai", {
        api_key: process.env.OPENAI_API_KEY,
        model: "gpt-4o",
      });
      for await (const resp of rt.callIterMethod("oai", "infer", {
        messages: [
          {
            role: "user",
            content:
              "Who is the president of US in 2021? Just answer in two words.",
          },
        ],
      })) {
        expect(resp.finish_reason).to.be.equal("stop");
        expect(resp.message.role).to.be.equal("assistant");
        expect(resp.message.content).to.contain("Joe Biden");
      }
      await rt.stop();
    });
  }
});
