import * as MCPClient from "@modelcontextprotocol/sdk/client/index.js";
import * as MCPClientStdio from "@modelcontextprotocol/sdk/client/stdio.js";

/**
 * MCPServer provides a high-level interface for interacting with an MCP stdio server
 * using the official MCP JavaScript SDK.
 *
 * - It manages a single `MCPClient.Client` instance and connects via `StdioClientTransport`.
 * - The connection lifecycle is handled through the `start()` and `cleanup()` methods.
 * - Tools can be discovered with `listTools()`, and invoked using `callTool()`.
 * - Unlike in Python, this class does not use subprocesses; it runs entirely within the calling Node.js process.
 */
class MCPServer {
  readonly name: string;
  #params: MCPClientStdio.StdioServerParameters;
  #client: MCPClient.Client;
  #connected: boolean = false;

  constructor(name: string, params: MCPClientStdio.StdioServerParameters) {
    this.name = name;
    this.#params = { ...params, stderr: "pipe" };
    this.#client = new MCPClient.Client({
      name,
      version: "dummy-version",
    });
  }

  async start() {
    if (this.#connected) return;
    const transport = new MCPClientStdio.StdioClientTransport(this.#params);

    let stderrOutput = "";

    transport.stderr!.on("data", (chunk: Buffer | string) => {
      stderrOutput += chunk.toString();
    });

    try {
      await this.#client.connect(transport);
      this.#connected = true;
    } catch (err) {
      await this.#client.close();
      throw Error(
        "Failed to start MCP server. Check the error output below.\n\n" +
          stderrOutput.trim()
      );
    }
  }

  async cleanup() {
    if (!this.#connected) return;
    await this.#client.close();
    this.#connected = false;
  }

  async listTools() {
    if (!this.#connected) {
      throw Error("MCP server not initialized");
    }

    const { tools } = await this.#client.listTools();
    return tools;
  }

  async callTool(
    tool: Awaited<ReturnType<MCPClient.Client["listTools"]>>["tools"][number],
    inputs?: { [x: string]: unknown }
  ) {
    try {
      const result = await this.#client.callTool({
        name: tool.name,
        arguments: inputs,
      });
      const content = result.content as Array<any>;
      const parsedContent = content.map((item) => {
        if (item.type === "text") {
          // Text Content
          try {
            // Try to deserialize as JSON
            const parsed = JSON.parse(item.text);
            return JSON.stringify(parsed);
          } catch (err) {
            // Return as-is if not deserializable
            return item.text;
          }
        } else if (item.type === "image") {
          // Image Content
          return item.data;
        } else if (item.type === "resource") {
          // Resource Content
          if (item.resource.text !== undefined) {
            // Text Resource
            return item.resource.text;
          } else {
            // Blob Resource
            return item.resource.blob;
          }
        }
      });
      return parsedContent;
    } catch (err) {
      console.error(`Error executing tool ${tool.name}: ${err}`);
      throw err;
    }
  }
}

export default MCPServer;
