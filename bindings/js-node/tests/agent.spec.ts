import { defineAgent } from "../src/agent";
import type { AgentResponse } from "../src/agent";
import { startRuntime, type Runtime } from "../src/runtime";

function printAgentResponse(resp: AgentResponse) {
  if (resp.type === "output_text") {
    process.stdout.write(resp.content);
    if (resp.endOfTurn) {
      process.stdout.write("\n\n");
    }
  } else if (resp.type === "tool_call") {
    process.stdout.write(`
  Tool Call
  - ID: ${resp.content.id}
  - name: ${resp.content.function.name}
  - arguments: ${JSON.stringify(resp.content.function.arguments)}
  `);
  } else if (resp.type === "tool_call_result") {
    process.stdout.write(`
  Tool Call Result
  - ID: ${resp.content.tool_call_id}
  - name: ${resp.content.name}
  - Result: ${resp.content.content}
  `);
  }
}

describe("Agent", async () => {
  let rt: Runtime;
  before(async () => {
    rt = await startRuntime("inproc://");
  });

  it("Tool Call: frankfurter tools", async () => {
    const agent = await defineAgent(rt, "qwen3-8b");
    agent.addToolsFromPreset("frankfurter");

    const query =
      "I want to buy 100 U.S. Dollar with my Korean Won. How much do I need to take?";
    process.stdout.write(`\nQuery: ${query}`);

    process.stdout.write(`\nAssistant: `);
    for await (const resp of agent.query(query)) {
      printAgentResponse(resp);
    }

    await agent.delete();
  });

  it("Tool Call: Custom function tools", async () => {
    const agent = await defineAgent(rt, "qwen3-8b");
    agent.addJSFunctionTool(
      {
        name: "get_current_temperature",
        description: "Get the current temperature at a location.",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description:
                'The location to get the temperature for, in the format "City, Country"',
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
              description: "The unit to return the temperature in.",
            },
          },
          required: ["location", "unit"],
        },
        return: {
          type: "number",
          description:
            "The current temperature at the specified location in the specified units, as a float.",
        },
      },
      ({ location, unit }) => {
        if (unit === "celsius") return 25;
        else if (unit === "fahrenheit") return 77;
        else return null;
      }
    );

    const query = "Hello, how is the current weather in my city Seoul?";
    process.stdout.write(`\nQuery: ${query}`);

    process.stdout.write(`\nAssistant: `);
    for await (const resp of agent.query(query)) {
      printAgentResponse(resp);
    }

    await agent.delete();
  });

  it("Tool Call: Github MCP tools", async () => {
    const agent = await defineAgent(rt, "qwen3-8b");
    await agent.addToolsFromMcpServer(
      {
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
      },
      {
        toolsToAdd: ["search_repositories", "get_file_contents"],
      }
    );

    const query =
      "Search the repository named brekkylab/ailoy, and describe what this repo does based on its README.md";
    process.stdout.write(`\nQuery: ${query}`);

    process.stdout.write(`\nAssistant: `);
    for await (const resp of agent.query(query)) {
      printAgentResponse(resp);
    }

    await agent.delete();
  });

  after(async () => {
    await rt.stop();
  });
});
