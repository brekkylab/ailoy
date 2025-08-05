import { describe, it, afterAll, expect } from "vitest";

import { defineAgent } from "../src/agent";
import { Runtime } from "../src/runtime";
import { APIModel } from "../src/models";
import { MCPStreamableHTTPClientTransport } from "../src/mcp";

describe("Agent MCP features", async () => {
  const runtime = new Runtime();
  await runtime.start();

  const agent = await defineAgent(
    runtime,
    APIModel({ id: "gpt-4o", apiKey: "<OPENAI_API_KEY>" })
  );

  it("Streamable HTTP Client", async () => {
    await agent.addToolsFromMcpClient(
      "test",
      new MCPStreamableHTTPClientTransport(new URL("http://localhost:3001/mcp"))
    );
    const tools = agent.getTools();
    expect(tools).to.be.length(2);

    const forecastTool = tools[1];
    let toolResult = await forecastTool.call({
      latitude: 32.7767,
      longitude: -96.797,
    });
    expect(toolResult).to.be.length(1);
    toolResult = toolResult[0];
    expect(toolResult).toMatch(/Forecast for 32.7767, -96.797/);
  });

  afterAll(async () => {
    await agent.delete();
    await runtime.stop();
  });
});
