import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";
import dts from "vite-plugin-dts";
import { viteStaticCopy } from "vite-plugin-static-copy";
import { transform } from "esbuild";

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
        external: ["./ailoy_js_web.js"],
        onwarn(warning, warn) {
          // Suppress eval-related warnings from wasm-vips
          if (
            warning.code === "EVAL" &&
            warning.loc !== undefined &&
            warning.loc.file !== undefined &&
            /node_modules\/wasm-vips/.test(warning.loc.file)
          ) {
            return;
          }

          // Log other warnings using the default handler
          warn(warning);
        },
      },
    },
    optimizeDeps: {
      exclude: ["wasm-vips"],
    },
    plugins: [
      dts({
        rollupTypes: true,
      }),
      minifyEs(),
      viteStaticCopy({
        targets: [
          {
            src: ["./src/ailoy_js_web.js", "./src/ailoy_js_web.wasm"],
            dest: "./",
          },
        ],
      }),
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
      testTimeout: 60000,
    },
    server: {
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      },
    },
  };
});

// Workaround for https://github.com/vitejs/vite/issues/6555
function minifyEs() {
  return {
    name: "minifyEs",
    renderChunk: {
      order: "post" as const,
      async handler(code, chunk, outputOptions) {
        if (outputOptions.format === "es") {
          return await transform(code, { minify: true });
        }
        return code;
      },
    },
  };
}
