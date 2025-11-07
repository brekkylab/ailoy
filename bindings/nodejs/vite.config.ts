import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import { viteStaticCopy } from "vite-plugin-static-copy";

const env = process.env.NODE_ENV;

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
    plugins: [
      dts({ rollupTypes: true }),
      env === "development" &&
        viteStaticCopy({
          targets: [
            {
              src: ["./src/ailoy_core.*.node"],
              dest: "./",
            },
          ],
        }),
    ],
  };
});
