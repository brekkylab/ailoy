import { assert, describe, expect, it } from "vitest";

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

  it("BuiltinTool: web_search_duckduckgo", async () => {
    // Creating tool without `base_url` should raise the error
    try {
      ailoy.Tool.newBuiltin("web_search_duckduckgo");
    } catch (err) {
      assert.include((err as Error).message, "not available");
    }

    // Creating tool with `base_url` pointing to proxy server
    const tool = ailoy.Tool.newBuiltin("web_search_duckduckgo", {
      base_url: "http://localhost:3001",
    });
    const results = await tool.run({
      query: "Ailoy",
      max_results: 5,
    });
    console.log(results);
  });

  it("BuiltinTool: web_fetch", async () => {
    const tool = ailoy.Tool.newBuiltin("web_fetch");
    const { results } = await tool.run({
      url: "https://brekkylab.github.io/ailoy/",
    });
    console.log(results);
  });
});
