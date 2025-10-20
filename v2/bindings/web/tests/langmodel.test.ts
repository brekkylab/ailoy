import { describe, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy LangModel", async () => {
  it("Local(Qwen3-0.6B)", async () => {
    const model = await ailoy.LangModel.newLocal("Qwen/Qwen3-0.6B", (prog) => {
      console.log(prog);
    });
    const msg: ailoy.Message = {
      role: "user",
      contents: [{ type: "text", text: "What is your name?" }],
    };
    for await (const result of model.infer([msg])) {
      console.log(result);
    }
  });

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
  it.skipIf(process.env.OPENAI_API_KEY === "undefined")("OpenAI", async () => {
    const model = await ailoy.LangModel.newStreamAPI(
      "OpenAI",
      "gpt-4o",
      OPENAI_API_KEY
    );
    const msg: ailoy.Message = {
      role: "user",
      contents: [{ type: "text", text: "What is your name?" }],
    };
    for await (const result of model.infer([msg])) {
      console.log(result);
    }
  });
});
