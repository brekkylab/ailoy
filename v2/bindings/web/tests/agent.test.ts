import { describe, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy Agent", async () => {
  it("Local(Qwen3-0.6B)", async () => {
    const model = await ailoy.LangModel.newLocal("Qwen/Qwen3-0.6B", (prog) => {
      console.log(prog);
    });
    const agent = new ailoy.Agent(model, []);
    for await (const response of agent.run([
      {
        role: "user",
        contents: [{ type: "text", text: "What is your name?" }],
      },
    ])) {
      console.log(response);
    }
  });

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
  it.skipIf(process.env.OPENAI_API_KEY === "undefined")("OpenAI", async () => {
    const model = await ailoy.LangModel.newStreamAPI(
      "OpenAI",
      "gpt-4o",
      OPENAI_API_KEY
    );

    const tool = ailoy.Tool.newFunction(
      {
        name: "temperature",
        description: "Get temperature for the given city",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city name",
            },
            unit: {
              type: "string",
              enum: ["Celsius", "Fahrenheit"],
            },
          },
          required: ["location", "unit"],
        },
        returns: {
          type: "number",
          description: "Null if the given city name is unavailable.",
          nullable: true,
        },
      },
      async (args: { location: string; unit: string }) => {
        return {
          location: args.location,
          unit: args.unit,
          temperature: 38,
        };
      }
    );

    const agent = new ailoy.Agent(model, [tool]);

    for await (const response of agent.run([
      {
        role: "user",
        contents: [
          {
            type: "text",
            text: "What is the temperature of Seoul in Celsius? Answer by using `temperature` tool.",
          },
        ],
      },
    ])) {
      console.log(response);
    }
  });
});
