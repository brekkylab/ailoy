import { Server } from "http";

import webSearchProxy from "./proxy/web-search";

let proxyServers: {
  webSearch: Server | undefined;
} = {
  webSearch: undefined,
};

export async function setup() {
  if (proxyServers.webSearch === undefined) {
    proxyServers.webSearch = webSearchProxy.listen(3001);
  }
}

export async function teardown() {
  if (proxyServers.webSearch !== undefined) {
    proxyServers.webSearch.close();
    proxyServers.webSearch = undefined;
  }
}
