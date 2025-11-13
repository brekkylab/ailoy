import os from "node:os";
import { defineConfig } from "vite";

process.stdout.write(`Platform: ${os.platform()}`);

// https://vitejs.dev/config
export default defineConfig({
  build: {
    target: "node20",
    rollupOptions: {
      external: [
        "ailoy-node",
        "ailoy-node-darwin-arm64",
        "ailoy-node-linux-x64-gnu",
        "ailoy-node-win32-x64-msvc",
      ],
    },
  },
});
