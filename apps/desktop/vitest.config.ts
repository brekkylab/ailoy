import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  // Node, because all but one of these tests are pure logic and a DOM would only be slower.
  // The exception says so in its own header (`// @vitest-environment jsdom`) and is a `.tsx`,
  // which is why the pattern takes both.
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
});
