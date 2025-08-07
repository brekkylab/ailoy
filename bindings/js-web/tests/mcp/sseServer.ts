import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import cors from "cors";
import createMcpServer from "./common";

const mcpServer = createMcpServer();

const app = express();
app.use(
  cors({
    credentials: true,
    exposedHeaders: "*",
  })
);

let transport: SSEServerTransport;

app.get("/sse", async (req, res) => {
  transport = new SSEServerTransport("/messages", res);
  await mcpServer.connect(transport);
});

app.post("/messages", async (req, res) => {
  await transport.handlePostMessage(req, res);
});

export default app;
