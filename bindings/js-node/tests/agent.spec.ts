import { createAgent } from "../src/agent";
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
    const ex = await createAgent(rt, {
      model: {
        name: "qwen3-8b",
      },
    });
    ex.addToolsFromPreset("frankfurter");

    const query =
      "I want to buy 100 U.S. Dollar with my Korean Won. How much do I need to take?";
    process.stdout.write(`\nQuery: ${query}`);

    process.stdout.write(`\nAssistant: `);
    for await (const resp of ex.run(query)) {
      printAgentResponse(resp);
    }

    await ex.deinitialize();
  });

  it("Tool Call: Github MCP tools", async () => {
    const ex = await createAgent(rt, {
      model: {
        name: "qwen3-8b",
      },
    });
    await ex.addToolsFromMcpServer(
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
    for await (const resp of ex.run(query)) {
      printAgentResponse(resp);
    }

    await ex.deinitialize();
  });

  after(async () => {
    await rt.stop();
  });
});
