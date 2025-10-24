import { beforeAll, describe, it, test } from "vitest";

import * as ailoy from "../src/index";

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

    test.sequential("Simple Chat", async () => {
      for await (const resp of agent.run([
        {
          role: "user",
          contents: [{ type: "text", text: "What is your name?" }],
        },
      ])) {
        if (resp.accumulated !== undefined) {
          console.log(resp.accumulated);
        }
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

        for await (const resp of agent.run([
          {
            role: "user",
            contents: [
              {
                type: "text",
                text: "What is the temperature in Seoul now? Answer in Celsius.",
              },
            ],
          },
        ])) {
          if (resp.accumulated !== undefined) {
            console.log(resp.accumulated);
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
            if (resp.accumulated !== undefined) {
              console.log(resp.accumulated);
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
          for await (const resp of agent.run([
            {
              role: "user",
              contents: [
                testImageBase64,
                { type: "text", text: "What is shown in this image?" },
              ],
            },
          ])) {
            if (resp.accumulated !== undefined) {
              console.log(resp.accumulated);
            }
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

        const documentPolyfill: ailoy.DocumentPolyfill = {
          systemMessageTemplate: `
    {{- text }}
    # Knowledges
    After the user’s question, a list of documents retrieved from the knowledge base may appear. Try to answer the user’s question based on the provided knowledges.
                `,
          queryMessageTemplate: `
    {{- text }}
    {%- if documents %}
        {{- "<documents>\n" }}
        {%- for doc in documents %}
        {{- "<document>\n" }}
            {{- doc.text + '\n' }}
        {{- "</document>\n" }}
        {%- endfor %}
        {{- "</documents>\n" }}
    {%- endif %}
                `,
        };

        for await (const resp of agent.run(
          [
            {
              role: "user",
              contents: [
                {
                  type: "text",
                  text: "What is Ailoy?",
                },
              ],
            },
          ],
          {
            documentPolyfill,
          }
        )) {
          if (resp.accumulated !== undefined) {
            console.log(resp.accumulated);
          }
        }

        agent.removeKnowledge();
      },
      120000
    );
  });
}
