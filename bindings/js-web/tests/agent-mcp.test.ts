import {
  describe,
  it,
  afterAll,
  expect,
  beforeEach,
  expectTypeOf,
} from "vitest";

import { defineAgent } from "../src/agent";
import { Runtime } from "../src/runtime";
import { APIModel } from "../src/models";
import {
  MCPClientTransport,
  MCPStreamableHTTPClientTransport,
  MCPSSEClientTransport,
  MCPWebSocketClientTransport,
} from "../src/mcp";

describe("Agent MCP features", async () => {
  const runtime = new Runtime();
  await runtime.start();

  const agent = await defineAgent(
    runtime,
    APIModel({ id: "gpt-4o", apiKey: "<OPENAI_API_KEY>" })
  );

  const testMcpTools = async (transport: MCPClientTransport) => {
    await agent.addToolsFromMcpClient("test", transport);
    const tools = agent.getTools();
    expect(tools).to.be.length(2);

    const forecastTool = tools[1];
    let toolResult = await forecastTool.call({
      latitude: 32.7767,
      longitude: -96.797,
    });
    expect(toolResult).to.be.length(1);
    toolResult = JSON.parse(toolResult[0]);
    expect(toolResult.location).to.be.equal("Dallas, TX");
    expectTypeOf(toolResult.periods).toMatchTypeOf<{
      detailedForecast: string;
      endTime: string;
      name: string;
      shortForecast: string;
      startTime: string;
      temperature: number;
      temperatureUnit: "F";
      windDirection: string;
      windSpeed: string;
    }>();

    await agent.removeMcpClient("test");
  };

  it("Streamable HTTP Client", async () => {
    await testMcpTools(
      new MCPStreamableHTTPClientTransport(
        new URL("http://localhost:3001/streamable-http")
      )
    );
  });

  it("SSE Client", async () => {
    await testMcpTools(
      new MCPSSEClientTransport(new URL("http://localhost:3002/sse"))
    );
  });

  it("WebSocket Client", async () => {
    await testMcpTools(
      new MCPWebSocketClientTransport(new URL("ws://localhost:3003/ws"))
    );
  });

  afterAll(async () => {
    await agent.delete();
    await runtime.stop();
  });
});
