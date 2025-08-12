import { Server } from "http";

import mcpSSEServer from "./mcp/sseServer";
import mcpCreateStreamableHttpServer from "./mcp/streamableHttpServer";
import mcpCreateWebsocketServer from "./mcp/websocketServer";

let mcpServers: {
  streamableHttp: Server | undefined;
  sse: Server | undefined;
  websocket: Server | undefined;
} = {
  streamableHttp: undefined,
  sse: undefined,
  websocket: undefined,
};

export async function setup() {
  mcpServers.streamableHttp = await new Promise(async (resolve) => {
    const instance = await mcpCreateStreamableHttpServer(3001, () =>
      resolve(instance)
    );
  });
  mcpServers.sse = await new Promise((resolve) => {
    const instance = mcpSSEServer.listen(3002, () => {
      resolve(instance);
    });
  });
  mcpServers.websocket = await new Promise(async (resolve) => {
    const instance = await mcpCreateWebsocketServer(3003);
    resolve(instance);
  });
}

export async function teardown() {
  if (mcpServers.streamableHttp !== undefined) {
    await new Promise((resolve) => mcpServers.streamableHttp?.close(resolve));
    mcpServers.streamableHttp = undefined;
  }
  if (mcpServers.sse !== undefined) {
    await new Promise((resolve) => mcpServers.sse?.close(resolve));
    mcpServers.sse = undefined;
  }
  if (mcpServers.websocket !== undefined) {
    await new Promise((resolve) => mcpServers.websocket?.close(resolve));
    mcpServers.websocket = undefined;
  }
}
