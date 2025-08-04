import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";
import { viteStaticCopy } from "vite-plugin-static-copy";

const SAFARI = process.env.BROWSER === "safari";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    define: {
      "process.env.OPENAI_API_KEY": JSON.stringify(env.OPENAI_API_KEY),
    },
    plugins: [
      viteStaticCopy({
        targets: [
          {
            src: [
              "node_modules/wasm-vips/lib/vips-es6.js",
              "node_modules/wasm-vips/lib/vips.wasm",
              "node_modules/wasm-vips/lib/vips-jxl.wasm",
              "node_modules/wasm-vips/lib/vips-heif.wasm",
            ],
            dest: "./",
          },
        ],
      }),
    ],
    optimizeDeps: {
      exclude: ["wasm-vips"],
    },
    test: {
      exclude: ["**/node_modules/**"],
      include: ["**/tests/*.test.**"],
      browser: {
        enabled: true,
        name: process.env.BROWSER ?? "chromium",
        provider: SAFARI ? "webdriverio" : "playwright",
        providerOptions: SAFARI
          ? {
              capabilities: {
                alwaysMatch: { browserName: "safari" },
                firstMatch: [{}],
                browserName: "safari",
              },
            }
          : {
              capabilities: {
                "goog:chromeOptions": {
                  args: ["no-sandbox", "disable-setuid-sandbox"],
                },
              },
            },
      },
      typecheck: {
        enabled: true,
      },
    },
    server: {
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      },
      hmr: false,
    },
  };
});
