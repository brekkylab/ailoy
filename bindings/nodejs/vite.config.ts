import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({}) => {
  return {
    build: {
      lib: {
        entry: ["src/index.ts"],
        formats: ["cjs"],
      },
      ssr: true,
      target: "node20",
      outDir: "dist",
    },
    plugins: [dts({ rollupTypes: true })],
  };
});
