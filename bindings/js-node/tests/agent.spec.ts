import { expect } from "chai";
import fs from "node:fs";
import path from "node:path";

import { defineAgent } from "../src/agent";
import { startRuntime, type Runtime } from "../src/runtime";

describe("Agent", async () => {
  let rt: Runtime;
  before(async () => {
    rt = await startRuntime("inproc://");
  });

  it("Tool Call: calculator tools", async () => {
    const agent = await defineAgent(rt, "Qwen/Qwen3-8B");
    agent.addToolsFromPreset("calculator");

    const query = "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))";
    process.stdout.write(`Query: ${query}`);

    for await (const resp of agent.query(query)) {
      agent.print(resp);
    }

    await agent.delete();
  });

  it("Tool Call: frankfurter tools", async () => {
    const agent = await defineAgent(rt, "Qwen/Qwen3-8B");
    agent.addToolsFromPreset("frankfurter");

    const query =
      "I want to buy 100 U.S. Dollar and 100 EUR with my Korean Won. How much do I need to take?";
    console.log(`Query: ${query}`);

    for await (const resp of agent.query(query)) {
      agent.print(resp);
    }

    await agent.delete();
  });

  it("Tool Call: Custom function tools", async () => {
    const agent = await defineAgent(rt, "Qwen/Qwen3-8B");
    agent.addJSFunctionTool(
      ({ location, unit }) => {
        if (unit === "celsius") return 25;
        else if (unit === "fahrenheit") return 77;
        else return null;
      },
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
      }
    );

    const query = "Hello, how is the current weather in my city Seoul?";
    console.log(`Query: ${query}`);

    for await (const resp of agent.query(query)) {
      agent.print(resp);
    }

    await agent.delete();
  });

  it("Tool Call: Filesystem MCP tools", async () => {
    const agent = await defineAgent(rt, "Qwen/Qwen3-8B");
    const testPath = path.join(__dirname, "..");
    await agent.addToolsFromMcpServer("filesystem", {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", testPath],
    });

    const query = `Create a file hello.txt under ${testPath} and write 'hello world'.`;
    console.log(`Query: ${query}`);

    for await (const resp of agent.query(query)) {
      agent.print(resp);
    }

    expect(fs.existsSync(`${testPath}/hello.txt`)).to.be.equal(true);
    expect(
      fs.readFileSync(`${testPath}/hello.txt`, { encoding: "utf-8" })
    ).to.be.equal("hello world");

    fs.unlinkSync(`${testPath}/hello.txt`);

    await agent.delete();
  });

  after(async () => {
    await rt.stop();
  });
});
