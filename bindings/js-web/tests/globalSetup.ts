import { Server } from "http";

import mcpStreamableHttpServer from "./mcp/streamableHttpServer";

let mcpServers: {
  streamableHttp: Server | undefined;
} = {
  streamableHttp: undefined,
};

export async function setup() {
  mcpServers.streamableHttp = await new Promise((resolve) => {
    const instance = mcpStreamableHttpServer.listen(3001, () => {
      resolve(instance);
    });
  });
}

export async function teardown() {
  if (mcpServers.streamableHttp !== undefined) {
    await new Promise((resolve) => mcpServers.streamableHttp?.close(resolve));
  }
}
