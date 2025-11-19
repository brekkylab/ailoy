import { beforeAll, describe, it, test } from "vitest";

import * as ailoy from "../src/index";

const modelConfigs = [
  {
    name: "Local",
    createAgent: async () => {
      const model = await ailoy.LangModel.newLocal("Qwen/Qwen3-0.6B", {
        progressCallback: (prog) => console.log(prog),
      });
      return new ailoy.Agent(model);
    },
  },
  {
    name: "OpenAI",
    skip: process.env.OPENAI_API_KEY === undefined,
    createAgent: async () => {
      const model = await ailoy.LangModel.newStreamAPI(
        "OpenAI",
        "gpt-4o",
        process.env.OPENAI_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageUrl: true,
    runImageBase64: true,
  },
  {
    name: "Gemini",
    skip: process.env.GEMINI_API_KEY === undefined,
    createAgent: async () => {
      const model = await ailoy.LangModel.newStreamAPI(
        "Gemini",
        "gemini-2.5-flash",
        process.env.GEMINI_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageBase64: true,
  },
  {
    name: "Claude",
    skip: process.env.ANTHROPIC_API_KEY === undefined,
    createAgent: async () => {
      const model = await ailoy.LangModel.newStreamAPI(
        "Claude",
        "claude-haiku-4-5",
        process.env.ANTHROPIC_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageUrl: true,
    runImageBase64: true,
  },
  {
    name: "Grok",
    skip: process.env.XAI_API_KEY === undefined,
    createAgent: async () => {
      const model = await ailoy.LangModel.newStreamAPI(
        "Grok",
        "grok-4-fast",
        process.env.XAI_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageUrl: true,
    runImageBase64: true,
  },
];

const testImageUrl =
  "https://newsroom.haas.berkeley.edu/wp-content/uploads/2023/02/jensen-huang-headshot2_thmb-300x246.jpg";
const testImageBase64 = await (async () => {
  const resp = await fetch(testImageUrl);
  const arrayBuffer = await resp.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  return ailoy.imageFromBytes(bytes);
})();

for (const cfg of modelConfigs) {
  describe.skipIf(cfg.skip).concurrent(`Ailoy Agent: ${cfg.name}`, () => {
    let agent: ailoy.Agent;

    beforeAll(async () => {
      agent = await cfg.createAgent();
    });

    test.sequential("Simple Chat(single string)", async () => {
      for await (const resp of agent.run("What is your name?")) {
        console.log(resp.message);
      }
    });

    test.sequential("Simple Chat(string contents)", async () => {
      for await (const resp of agent.run([
        { role: "user", contents: "What is your name?" },
      ])) {
        console.log(resp.message);
      }
    });

    test.sequential("Simple Chat(normal form)", async () => {
      for await (const resp of agent.run([
        {
          role: "user",
          contents: [{ type: "text", text: "What is your name?" }],
        },
      ])) {
        console.log(resp.message);
      }
    });

    test.sequential("Simple Chat Delta", async () => {
      let acc = { contents: [], tool_calls: [] } as ailoy.MessageDelta;
      for await (const resp of agent.runDelta([
        {
          role: "user",
          contents: [{ type: "text", text: "What is your name?" }],
        },
      ])) {
        acc = ailoy.accumulateMessageDelta(acc, resp.delta);
      }
      const message = ailoy.finishMessageDelta(acc);
      if (message.contents?.[0]?.type === "text") {
        console.log(message.contents[0].text);
      }
    });

    test.sequential(
      "Tool Calling: JsFunctionTool (temperature)",
      async () => {
        const tool = ailoy.Tool.newFunction(
          {
            name: "temperature",
            description: "Get temperature of the provided location",
            parameters: {
              type: "object",
              properties: {
                location: {
                  type: "string",
                  description: "The city name",
                },
                unit: {
                  type: "string",
                  description: "temperature unit",
                  enum: ["Celsius", "Fahrenheit"],
                },
              },
              required: ["location", "unit"],
            },
          },
          async (args) => {
            return {
              location: args.location,
              unit: args.unit,
              temp: 38,
            };
          }
        );

        agent.addTool(tool);

        for await (const resp of agent.run(
          "What is the temperature in Seoul now? Answer in Celsius."
        )) {
          console.log(resp.message);
        }

        agent.removeTool(tool.description.name);
      },
      10000
    );

    if (cfg.runImageUrl) {
      test.sequential(
        "Multimodal: Image URL",
        async () => {
          const imgPart = ailoy.imageFromUrl(testImageUrl);
          for await (const resp of agent.run([
            {
              role: "user",
              contents: [
                imgPart,
                { type: "text", text: "What is shown in this image?" },
              ],
            },
          ])) {
            console.log(resp.message);
          }
        },
        10000
      );
    }

    if (cfg.runImageBase64) {
      test.sequential(
        "Multimodal: Image Base64",
        async () => {
          for await (const resp of agent.run([
            {
              role: "user",
              contents: [
                testImageBase64,
                { type: "text", text: "What is shown in this image?" },
              ],
            },
          ])) {
            console.log(resp.message);
          }
        },
        10000
      );
    }

    test.sequential(
      "Using Knowledge",
      async () => {
        const vs = await ailoy.VectorStore.newFaiss(1024);
        const emb = await ailoy.EmbeddingModel.newLocal("BAAI/bge-m3");

        const doc0 =
          "Ailoy is an awesome AI agent framework supporting Rust, Python, Nodejs and WebAssembly.";
        const emb0 = await emb.infer(doc0);
        await vs.addVector({ embedding: emb0, document: doc0 });

        const knowledge = ailoy.Knowledge.newVectorStore(vs, emb);
        agent.setKnowledge(knowledge);

        const documentPolyfill = ailoy.getDocumentPolyfill("Qwen3");
        for await (const resp of agent.run("What is Ailoy?", {
          inference: { documentPolyfill },
        })) {
          console.log(resp.message);
        }

        agent.removeKnowledge();
      },
      120000
    );
  });
}
