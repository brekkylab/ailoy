import os from "node:os";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

process.stdout.write(`Platform: ${os.platform()}`);

const targetsToCopy: string[] = [
  "node_modules/ailoy-node/dist/ailoy_addon.node",
];
if (os.platform() === "win32") {
  targetsToCopy.push(
    "node_modules/ailoy-node/dist/tvm_runtime.dll",
    "node_modules/ailoy-node/dist/mlc_llm.dll"
  );
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
