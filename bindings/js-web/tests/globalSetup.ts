import { Server } from "http";

import mcpStreamableHttpServer from "./mcp/streamableHttpServer";
import mcpSSEServer from "./mcp/sseServer";

let mcpServers: {
  streamableHttp: Server | undefined;
  sse: Server | undefined;
} = {
  streamableHttp: undefined,
  sse: undefined,
};

export async function setup() {
  mcpServers.streamableHttp = await new Promise((resolve) => {
    const instance = mcpStreamableHttpServer.listen(3001, () => {
      resolve(instance);
    });
  });
  mcpServers.sse = await new Promise((resolve) => {
    const instance = mcpSSEServer.listen(3002, () => {
      resolve(instance);
    });
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
}
