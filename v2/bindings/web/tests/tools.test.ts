import { describe, expect, it } from "vitest";

import * as ailoy from "../src/index";

describe("Ailoy Tools", async () => {
  it("FunctionTool", async () => {
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
        console.log(args.location);
        console.log(args.unit);
        return {
          location: args.location,
          unit: args.unit,
          temperature: 38,
        };
      }
    );

    let ret = await tool.run({ location: "Seoul", unit: "Celsius" });
    expect(ret).to.deep.equal({
      location: "Seoul",
      unit: "Celsius",
      temperature: 38,
    });
  });

  it("MCPTool: Streamable HTTP", async () => {
    const client = await ailoy.MCPClient.streamableHttp(
      "http://localhost:8123/mcp"
    );
    const tool = client.tools[1];
    const result = await tool.run({ latitude: 32.7767, longitude: -96.797 });
    console.log(result);
  });
});
