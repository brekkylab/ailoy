import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import cors from "cors";
import express, { type Request, type Response } from "express";
import * as z from "zod/v3";

const server = new McpServer({
  name: "demo-server",
  version: "1.0.0",
});

// @ts-ignore
server.registerTool(
  "add",
  {
    description: "Add two given numbers",
    inputSchema: {
      a: z.number().describe("The first number to add"),
      b: z.number().describe("The second number to add"),
    },
  },
  async ({ a, b }) => ({
    content: [{ type: "text", text: String(a + b) }],
  })
);

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: "*",
    exposedHeaders: "*",
  })
);

app.post("/mcp", async (req: Request, res: Response) => {
  try {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });
    res.on("close", () => {
      transport.close();
      server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`MCP server listening on ${PORT}`);
});
