import { Server } from "http";

import proxyDuckduckgo from "./proxy/duckduckgo";

let proxyServers: {
  duckduckgo: Server | undefined;
} = {
  duckduckgo: undefined,
};

export async function setup() {
  if (proxyServers.duckduckgo === undefined) {
    proxyServers.duckduckgo = proxyDuckduckgo.listen(3001);
  }
}

export async function teardown() {
  if (proxyServers.duckduckgo !== undefined) {
    proxyServers.duckduckgo.close();
    proxyServers.duckduckgo = undefined;
  }
}
