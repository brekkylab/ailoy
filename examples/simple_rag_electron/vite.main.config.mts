import os from "node:os";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// https://vitejs.dev/config
export default defineConfig({
  assetsInclude: ["**/*.node"],
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/ailoy-js-node/dist/ailoy_addon.node",
          dest: "./",
        },
        {
            src: os.platform() === "win32" ? ["node_modules/ailoy-js-node/dist/tvm_runtime.dll", "node_modules/ailoy-js-node/dist/mlc_llm.dll"] : [],
          dest: "./",
        },
      ],
    }),
  ],
});
