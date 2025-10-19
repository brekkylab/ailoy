import { loadEnv } from "vite";
import dts from "vite-plugin-dts";
import { viteStaticCopy } from "vite-plugin-static-copy";
import wasm from "vite-plugin-wasm";
import { defineConfig } from "vitest/config";

const SAFARI = process.env.BROWSER === "safari";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    build: {
      lib: {
        entry: ["src/index.ts"],
        formats: ["es"],
      },
      rollupOptions: {
        external: [
          "./ailoy-web_bg.js",
          "./ailoy-web.js",
          "./shim_js/dist/index.js",
        ],
      },
    },
    plugins: [
      dts({
        rollupTypes: true,
      }),
      viteStaticCopy({
        targets: [
          {
            src: [
              "./src/ailoy-web_bg.js",
              "./src/ailoy-web_bg.wasm",
              "./src/ailoy-web.js",
              "./src/shim_js",
            ],
            dest: "./",
          },
        ],
      }),
      wasm(),
    ],
    define: {
      "process.env.OPENAI_API_KEY": JSON.stringify(env.OPENAI_API_KEY),
      "process.env.GEMINI_API_KEY": JSON.stringify(env.GEMINI_API_KEY),
      "process.env.CLAUDE_API_KEY": JSON.stringify(env.CLAUDE_API_KEY),
      "process.env.XAI_API_KEY": JSON.stringify(env.XAI_API_KEY),
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
      testTimeout: 120000,
    },
    server: {
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      },
    },
  };
});
