import { describe, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy LangModel", async () => {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
  it.skipIf(process.env.OPENAI_API_KEY === "undefined")("OpenAI", async () => {
    const model = await ailoy.LangModel.create_stream_api(
      "OpenAI",
      "gpt-4o",
      OPENAI_API_KEY
    );
    const msg: ailoy.Message = {
      role: "user",
      contents: [{ type: "text", text: "What is your name?" }],
    };
    const iter = model.infer([msg]);
    for await (const result of iter) {
      console.log(result);
    }
  });
});
