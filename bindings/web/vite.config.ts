import { playwright } from "@vitest/browser-playwright";
import { webdriverio } from "@vitest/browser-webdriverio";
import { loadEnv } from "vite";
import dts from "vite-plugin-dts";
import { viteStaticCopy } from "vite-plugin-static-copy";
import wasm from "vite-plugin-wasm";
import { Plugin, defineConfig } from "vitest/config";

const SAFARI = process.env.BROWSER === "safari";

function rewriteImportPath(
  options: {
    mappings?: Record<string, string>;
  } = {}
): Plugin {
  const { mappings = {} } = options;

  function escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  return {
    name: "rewrite-import-path",
    enforce: "post",
    generateBundle(_, bundle) {
      // Iterate through all chunks in the bundle
      for (const fileName in bundle) {
        const chunk = bundle[fileName];

        // Only process JavaScript chunks
        if (chunk.type === "chunk" && fileName.endsWith(".js")) {
          let code = chunk.code;
          let modified = false;

          // Replace each mapping in the code
          for (const [from, to] of Object.entries(mappings)) {
            let escapedFrom = escapeRegExp(from);
            const patterns = [
              // import from "..."
              new RegExp(`from\\s+["']${escapedFrom}["']`, "g"),
              // import("...")
              new RegExp(`import\\s*\\(\\s*["']${escapedFrom}["']\\s*\\)`, "g"),
              // require("...")
              new RegExp(
                `require\\s*\\(\\s*["']${escapedFrom}["']\\s*\\)`,
                "g"
              ),
            ];

            for (const pattern of patterns) {
              const originalCode = code;
              code = code.replace(pattern, (match) => match.replace(from, to));
              if (code !== originalCode) {
                modified = true;
              }
            }
          }

          // Update the chunk code if modified
          if (modified) {
            chunk.code = code;
          }
        }
      }
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    build: {
      lib: {
        entry: ["src/index.ts"],
        formats: ["es"],
      },
      rollupOptions: {
        external: ["./shim_js/dist/index.js"],
      },
    },
    plugins: [
      dts({
        rollupTypes: true,
      }),
      viteStaticCopy({
        targets: [
          {
            src: ["./src/shim_js/dist/index.js"],
            dest: "./",
            rename: "shim.js",
          },
        ],
      }),
      wasm(),
      rewriteImportPath({
        mappings: {
          "./shim_js/dist/index.js": "./shim.js",
        },
      }),
    ],
    define: {
      "process.env.OPENAI_API_KEY": JSON.stringify(env.OPENAI_API_KEY),
      "process.env.GEMINI_API_KEY": JSON.stringify(env.GEMINI_API_KEY),
      "process.env.ANTHROPIC_API_KEY": JSON.stringify(env.ANTHROPIC_API_KEY),
      "process.env.XAI_API_KEY": JSON.stringify(env.XAI_API_KEY),
    },
    test: {
      exclude: ["**/node_modules/**"],
      include: ["**/tests/*.test.**"],
      browser: {
        enabled: true,
        name: process.env.BROWSER ?? "chromium",
        provider: SAFARI ? webdriverio() : playwright(),
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
        instances: [{ browser: "chromium" }],
      },
      globalSetup: "./tests/globalSetup.ts",
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
