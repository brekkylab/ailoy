import { describe, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy Agent", async () => {
  it("Local(Qwen3-0.6B)", async () => {
    const model = await ailoy.LangModel.create_local(
      "Qwen/Qwen3-0.6B",
      (prog) => {
        console.log(prog);
      }
    );
    const agent = new ailoy.Agent(model, []);
    for await (const response of agent.run([
      { type: "text", text: "What is your name?" },
    ])) {
      console.log(response);
    }
  });

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
  it.skipIf(process.env.OPENAI_API_KEY === "undefined")("OpenAI", async () => {
    const model = await ailoy.LangModel.create_stream_api(
      "OpenAI",
      "gpt-4o",
      OPENAI_API_KEY
    );
    const agent = new ailoy.Agent(model, []);
    for await (const response of agent.run([
      { type: "text", text: "What is your name?" },
    ])) {
      console.log(response);
    }
  });
});
