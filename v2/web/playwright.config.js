// playwright.config.js
import { defineConfig } from "@playwright/test";

export default defineConfig({
  webServer: {
    command: "npx serve . -l 3000",
    port: 3000,
    reuseExistingServer: true, // 이미 서버가 있으면 재사용
  },
  testDir: "./tests",
});
