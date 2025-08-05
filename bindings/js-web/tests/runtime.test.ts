import { describe, it, expect, beforeAll, afterAll } from "vitest";

import { Runtime } from "../src/runtime";

describe("Ailoy Runtime", async () => {
  const rt = new Runtime();
  await rt.start();

  it("FAISS Vectorstore", async () => {
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
    expect(resp2.embedding.getData()).to.deep.equal(insertInputs[0].embedding);
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
  });

  it("Text Split", async () => {
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
  });

  it("HTTP Request", async () => {
    const args = {
      url: "https://api.frankfurter.dev/v1/latest?base=USD&symbols=KRW",
      method: "GET",
      headers: {},
    };
    const resp = await rt.call("http_request", args);
    expect(resp.status_code).to.be.equal(200);
    expect(resp.headers["content-type"])
      .to.be.a("string")
      .and.satisfy((data: string) => data.startsWith("application/json"));

    const body = JSON.parse(new TextDecoder().decode(resp.body));
    expect(body.amount).to.be.equal(1.0);
    expect(body.base).to.be.equal("USD");
    expect(body.rates).to.have.property("KRW");
  });

  afterAll(async () => {
    await rt.stop();
  });
});

describe("Language-related Components", async () => {
  const rt = new Runtime();
  await rt.start();

  // Prepare OPFS directories
  let handle: FileSystemDirectoryHandle =
    await navigator.storage.getDirectory();
  const filepathPrefix = "tvm-models/Qwen--Qwen3-0.6B/q4f16_1";
  for (const dirname of ["ailoy", ...filepathPrefix.split("/")]) {
    handle = await handle.getDirectoryHandle(dirname, { create: true });
  }

  const chatMessages = [
    {
      role: "system",
      content: [{ type: "text", text: "You are awesome model." }],
    },
    {
      role: "user",
      content: [{ type: "text", text: "What is your name?" }],
    },
  ];
  const expectedTemplateResult = `<|im_start|>system
You are awesome model.<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
<think>

</think>

`;
  const expectedTokens = [
    151644, 8948, 198, 2610, 525, 12456, 1614, 13, 151645, 198, 151644, 872,
    198, 3838, 374, 697, 829, 30, 151645, 198, 151644, 77091, 198, 151667, 271,
    151668, 271,
  ];

  beforeAll(async () => {
    // Download required files
    for (const filename of [
      "chat-template-config.json",
      "Qwen--Qwen3-0.6B.j2",
      "tokenizer.json",
    ]) {
      const resp = await fetch(
        `https://models.download.ailoy.co/${filepathPrefix}/${filename}`
      );
      const writable = await (
        await handle.getFileHandle(filename, { create: true })
      ).createWritable();
      await writable.write(new Uint8Array(await resp.arrayBuffer()));
      await writable.close();
    }
  });

  it.sequential("Chat Manager", async () => {
    await rt.define("chat_manager", "cm0", {
      model: "Qwen/Qwen3-0.6B",
    });
    const templated = await rt.callMethod("cm0", "apply_chat_template", {
      messages: chatMessages,
      tools: [],
      reasoning: false,
      add_generation_prompt: true,
    });
    expect(templated.result).to.be.equal(expectedTemplateResult);

    await rt.delete("cm0");
  });

  it.sequential("Tokenizer", async () => {
    await rt.define("tokenizer", "t0", { model: "Qwen/Qwen3-0.6B" });

    const encoded = await rt.callMethod("t0", "encode", {
      text: expectedTemplateResult,
    });
    expect(encoded.tokens).to.deep.equal(expectedTokens);

    const decoded = await rt.callMethod("t0", "decode", {
      tokens: encoded.tokens,
    });
    expect(decoded.text).to.be.equal(expectedTemplateResult);

    await rt.delete("t0");
  });

  afterAll(async () => {
    await rt.stop();
  });
});
