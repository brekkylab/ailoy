import dotenv from "dotenv";
import { beforeAll, describe, test } from "vitest";

import * as ailoy from "../src";

dotenv.config();

const modelConfigs = [
  {
    name: "Local",
    createAgent: async () => {
      const model = await ailoy.LocalLanguageModel.create("Qwen/Qwen3-0.6B");
      model.disableReasoning();
      return new ailoy.Agent(model);
    },
  },
  {
    name: "OpenAI",
    skip: process.env.OPENAI_API_KEY === undefined,
    createAgent: async () => {
      const model = new ailoy.OpenAILanguageModel(
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
      const model = new ailoy.GeminiLanguageModel(
        "gemini-2.5-flash",
        process.env.GEMINI_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageBase64: true,
  },
  {
    name: "Anthropic",
    skip: process.env.ANTHROPIC_API_KEY === undefined,
    createAgent: async () => {
      const model = new ailoy.AnthropicLanguageModel(
        "claude-sonnet-4-20250514",
        process.env.ANTHROPIC_API_KEY!
      );
      return new ailoy.Agent(model);
    },
    runImageBase64: true,
  },
  {
    name: "XAI",
    skip: process.env.XAI_API_KEY === undefined,
    createAgent: async () => {
      const model = new ailoy.XAILanguageModel(
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
  return buffer.toString("base64");
})();

for (const cfg of modelConfigs) {
  describe.skipIf(cfg.skip).concurrent(`Agent test: ${cfg.name}`, () => {
    let agent: ailoy.Agent;

    beforeAll(async () => {
      agent = await cfg.createAgent();
    });

    test.sequential("Simple Chat", async () => {
      const agg = new ailoy.MessageAggregator();
      for await (const resp of agent.run("What is your name?")) {
        const msg = agg.update(resp);
        if (msg !== null) {
          console.log(msg);
        }
      }
    });

    test.sequential(
      "Tool Calling: Builtin Tool (terminal)",
      async () => {
        const tool = ailoy.BuiltinTool.terminal();
        agent.addTool(tool);

        const agg = new ailoy.MessageAggregator();
        for await (const resp of agent.run(
          "List the files in the current directory."
        )) {
          const msg = agg.update(resp);
          if (msg !== null) {
            console.log(msg);
          }
        }

        agent.removeTool(tool.description.name);
      },
      10000
    );

    test.sequential(
      "Tool Calling: MCP Tools (time)",
      async () => {
        const transport = ailoy.MCPTransport.newStdio("uvx", [
          "mcp-server-time",
        ]);
        const tools = await transport.tools("time");
        agent.addTools(tools);

        const agg = new ailoy.MessageAggregator();
        for await (const resp of agent.run(
          "What time is it now in Asia/Seoul? Answer in local timezone."
        )) {
          const msg = agg.update(resp);
          if (msg !== null) {
            console.log(msg);
          }
        }

        agent.removeTools(tools.map((t) => t.description.name));
      },
      10000
    );

    test.sequential(
      "Tool Calling: JsFunctionTool (temperature)",
      async () => {
        const tool = new ailoy.JsFunctionTool(
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

        const agg = new ailoy.MessageAggregator();
        for await (const resp of agent.run(
          "What is the temperature in Seoul now? Answer in Celsius."
        )) {
          const msg = agg.update(resp);
          if (msg !== null) {
            console.log(msg);
          }
        }

        agent.removeTool(tool.description.name);
      },
      10000
    );

    if (cfg.runImageUrl) {
      test.sequential(
        "Multimodal: Image URL",
        async () => {
          const imgPart = ailoy.Part.newImageUrl(testImageUrl);
          const agg = new ailoy.MessageAggregator();
          for await (const resp of agent.run([
            imgPart,
            "What is shown in this image?",
          ])) {
            const msg = agg.update(resp);
            if (msg !== null) {
              console.log(msg);
            }
          }
        },
        10000
      );
    }

    if (cfg.runImageBase64) {
      test.sequential(
        "Multimodal: Image Base64",
        async () => {
          const imgPart = ailoy.Part.newImageData(
            testImageBase64,
            "image/jpeg"
          );

          const agg = new ailoy.MessageAggregator();
          for await (const resp of agent.run([
            imgPart,
            "What is shown in this image?",
          ])) {
            const msg = agg.update(resp);
            if (msg !== null) {
              console.log(msg);
            }
          }
        },
        10000
      );
    }
  });
}
