// playwright.config.js
import { defineConfig } from "@playwright/test";

export default defineConfig({
  webServer: {
    command: "npx serve . -l 3000",
    port: 3000,
    reuseExistingServer: true,
  },
  testDir: "./tests",
});
