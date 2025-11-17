import dotenv from "dotenv";
import { beforeAll, describe, test } from "vitest";

import * as ailoy from "../src";

dotenv.config();

const modelConfigs = [
  {
    name: "Local",
    createAgent: async () => {
      const model = await ailoy.LangModel.newLocal("Qwen/Qwen3-0.6B");
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
  "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";
const testImageBase64 = await (async () => {
  const resp = await fetch(testImageUrl);
  const arrayBuffer = await resp.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const b64 = buffer.toString("base64");
  return ailoy.imageFromBase64(b64);
})();

for (const cfg of modelConfigs) {
  describe.skipIf(cfg.skip).concurrent(`Agent test: ${cfg.name}`, () => {
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
      let acc = {
        contents: [],
        tool_calls: [],
      } as ailoy.MessageDelta;
      for await (const resp of agent.runDelta([
        {
          role: "user",
          contents: [{ type: "text", text: "What is your name?" }],
        },
      ])) {
        acc = ailoy.accumulateMessageDelta(acc, resp.delta);
      }
      const message = ailoy.finishMessageDelta(acc);
      if (message.contents[0].type === "text")
        console.log(message.contents[0].text);
    });

    test.sequential(
      "Tool Calling: Builtin Tool (terminal)",
      async () => {
        const tool = ailoy.Tool.newBuiltin("terminal");
        agent.addTool(tool);

        for await (const resp of agent.runDelta(
          "List the files in the current directory."
        )) {
          console.log(`[${cfg.name}] `, resp.delta);
        }

        agent.removeTool(tool.description.name);
      },
      10000
    );

    test.sequential(
      "Tool Calling: MCP Tools (time)",
      async () => {
        const client = await ailoy.MCPClient.newStdio("uvx", [
          "mcp-server-time",
        ]);
        agent.addTools(client.tools);

        for await (const resp of agent.runDelta(
          "What time is it now in Asia/Seoul? Answer in local timezone."
        )) {
          console.log(resp.delta);
        }

        agent.removeTools(client.tools.map((t) => t.description.name));
      },
      10000
    );

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

        for await (const resp of agent.runDelta(
          "What is the temperature in Seoul now? Answer in Celsius."
        )) {
          console.log(resp.delta);
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
        for await (const resp of agent.runDelta("What is Ailoy?", {
          inference: { documentPolyfill },
        })) {
          console.log(resp.delta);
        }

        agent.removeKnowledge();
      },
      12000
    );
  });
}
