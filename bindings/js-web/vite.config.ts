import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";
import dts from "vite-plugin-dts";

const SAFARI = process.env.BROWSER === "safari";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    build: {
      lib: {
        entry: ["src/index.ts"],
        formats: ["es"],
      },
      minify: true,
      sourcemap: true,
      rollupOptions: {
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
    plugins: [
      dts({
        rollupTypes: true,
      }),
    ],
    optimizeDeps: {
      exclude: ["wasm-vips", "ailoy_js_web"],
    },
    define: {
      "process.env.OPENAI_API_KEY": JSON.stringify(env.OPENAI_API_KEY),
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
    },
  };
});
