import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { WebSocketClientTransport } from "@modelcontextprotocol/sdk/client/websocket.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { RequestOptions } from "@modelcontextprotocol/sdk/shared/protocol.js";
import {
  JSONRPCMessage,
  MessageExtraInfo,
} from "@modelcontextprotocol/sdk/types.js";

export type MCPClientTransport =
  | StreamableHTTPClientTransport
  | SSEClientTransport
  | WebSocketClientTransport;

export interface MCPClientStartOptions extends RequestOptions {
  onmessage?: (message: JSONRPCMessage, extra?: MessageExtraInfo) => void;
  onclose?: () => void;
  onerror?: (error: Error) => void;
}

/**
 * MCPClient provides a high-level interface for interacting with an internal MCP client
 * using the official MCP JavaScript SDK.
 *
 * - It manages a single `Client` instance and connects via provided transport (except `StdioClientTransport`).
 * - The connection lifecycle is handled through the `start()` and `cleanup()` methods.
 * - Tools can be discovered with `listTools()`, and invoked using `callTool()`.
 */
export class MCPClient {
  #client: Client;

  constructor(name: string) {
    this.#client = new Client({ name, version: "dummy-version" });
  }

  async isAlive() {
    try {
      await this.#client.ping();
      return true;
    } catch (e) {
      return false;
    }
  }

  async start(transport: MCPClientTransport, options?: MCPClientStartOptions) {
    if (await this.isAlive()) return;

    let errorOutput = "";

    transport.onmessage = options?.onmessage;
    transport.onclose = options?.onclose;
    transport.onerror = (error: Error) => {
      errorOutput = `${error.name}: ${error.message}`;
      options?.onerror?.(error);
    };

    try {
      await this.#client.connect(transport, options);
    } catch (err) {
      await this.#client.close();
      throw Error(
        "Failed to connect to MCP server. Check the error output below.\n\n" +
          errorOutput
      );
    }
  }

  async cleanup() {
    if (!(await this.isAlive())) return;
    await this.#client.close();
  }

  async listTools() {
    const { tools } = await this.#client.listTools();
    return tools;
  }

  async callTool(
    tool: Awaited<ReturnType<Client["listTools"]>>["tools"][number],
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

export {
  StreamableHTTPClientTransport as MCPStreamableHTTPClientTransport,
  SSEClientTransport as MCPSSEClientTransport,
  WebSocketClientTransport as MCPWebSocketClientTransport,
};
