import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import express from "express";
import cors from "cors";
import { WebSocket, WebSocketServer } from "ws";
import { IncomingMessage } from "http";
import { Readable, Writable } from "stream";

import createMcpServer from "./common";

const mcpServer = createMcpServer();

class WebSocketReadableStream extends Readable {
  constructor(private ws: WebSocket) {
    super({ objectMode: false });

    this.ws.on("message", (data: Buffer | string) => {
      const message = typeof data === "string" ? data : data.toString();
      const formattedMessage = message.endsWith("\n")
        ? message
        : message + "\n";
      this.push(Buffer.from(formattedMessage, "utf8"));
    });

    this.ws.on("close", () => {
      this.push(null);
    });

    this.ws.on("error", (error: Error) => {
      this.destroy(error);
    });
  }

  _read(): void {
    // No-op: data is pushed when WebSocket receives messages
  }
}

class WebSocketWritableStream extends Writable {
  constructor(private ws: WebSocket) {
    super({ objectMode: false });
  }

  _write(
    chunk: any,
    encoding: BufferEncoding,
    callback: (error?: Error | null) => void
  ): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      try {
        const message = chunk.toString().replace(/\n$/, "");
        this.ws.send(message, undefined, callback);
      } catch (error) {
        callback(error as Error);
      }
    } else {
      callback(new Error("WebSocket is not open"));
    }
  }

  _final(callback: (error?: Error | null) => void): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
    callback();
  }
}

async function createWebSocketServer(port: number) {
  const app = express();
  app.use(express.json());
  app.use(
    cors({
      credentials: true,
      exposedHeaders: "*",
    })
  );
  const httpServer = app.listen(port);

  const wss = new WebSocketServer({
    server: httpServer,
    path: "/ws",
  });

  wss.on("connection", (ws: WebSocket, request: IncomingMessage) => {
    console.log(
      `New Websocket connection from ${request.socket.remoteAddress}`
    );

    try {
      const readableStream = new WebSocketReadableStream(ws);
      const writableStream = new WebSocketWritableStream(ws);

      const transport = new StdioServerTransport(
        readableStream,
        writableStream
      );

      mcpServer.connect(transport).catch((error: Error) => {
        console.log("Failed to connect MCP server to transport:", error);
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1011, "Server error during MCP connection");
        }
      });

      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.ping();
        } else {
          clearInterval(pingInterval);
        }
      }, 30000);

      ws.on("close", (code: number, reason: Buffer) => {
        console.log(
          `WebSocket connection closed: ${code} ${reason.toString()}`
        );
        clearInterval(pingInterval);
        if (!readableStream.destroyed) readableStream.destroy();
        if (!writableStream.destroyed) writableStream.destroy();
      });

      ws.on("error", (error: Error) => {
        console.error("WebSocket error:", error);
        clearInterval(pingInterval);
        if (!readableStream.destroyed) readableStream.destroy();
        if (!writableStream.destroyed) writableStream.destroy();
      });
    } catch (error: unknown) {
      console.error("Error setting up WebSocket connection:", error);
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
  });

  wss.on("error", (error: Error) => {
    console.error("WebSocket server error:", error);
  });

  process.on("SIGINT", () => {
    console.log("\nShutting down server...");
    wss.close(() => {
      console.log("Server shut down gracefully");
      process.exit(0);
    });
  });

  return httpServer;
}

export default createWebSocketServer;
