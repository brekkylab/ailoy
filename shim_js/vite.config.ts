import { defineConfig } from "vitest/config";

const SAFARI = process.env.BROWSER === "safari";

export default defineConfig(({}) => {
  return {
    build: {
      lib: {
        entry: ["src/index.ts"],
        formats: ["es"],
      },
    },
    plugins: [],
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
        typecheck: {
          enabled: true,
        },
      },
    },
  };
});
