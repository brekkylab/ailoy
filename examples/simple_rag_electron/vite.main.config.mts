import os from "node:os";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

process.stdout.write(`Platform: ${os.platform()}`);

const targetsToCopy: string[] = [
  "node_modules/ailoy-node/dist/ailoy_addon.node",
];
if (os.platform() === "darwin") {
  targetsToCopy.push("node_modules/ailoy-node/dist/*.dylib");
} else if (os.platform() === "win32") {
  targetsToCopy.push("node_modules/ailoy-node/dist/*.dll");
} else if (os.platform() === "linux") {
  targetsToCopy.push("node_modules/ailoy-node/dist/*.so");
}

// https://vitejs.dev/config
export default defineConfig({
  assetsInclude: ["**/*.node"],
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: targetsToCopy,
          dest: "./",
        },
      ],
    }),
  ],
});
