import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import cors from "cors";
import express from "express";

import createMcpServer from "./common";

let transport: StreamableHTTPServerTransport;

const app = express();
app.use(express.json());
app.use(
  cors({
    credentials: true,
    exposedHeaders: "*",
  })
);

app.post("/streamable-http", async (req, res) => {
  console.log("New MCP streaming connection established");

  try {
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error("Error handling MCP request:", error);
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

async function createServer(port: number, callback: () => void) {
  const mcpServer = createMcpServer();
  transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  });
  await mcpServer.connect(transport);
  return app.listen(port, callback);
}

export default createServer;
