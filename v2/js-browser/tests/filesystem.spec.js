import { test, expect } from "@playwright/test";
import { resolve as _resolve } from "path";

test.describe("Filesystem integration", () => {
  test("should write and read file from OPFS", async ({ page }) => {
    // Register log handler
    page.on("console", (msg) => {
      console.log(`[browser:${msg.type()}] ${msg.text()}`);
    });

    // Serve static WASM module
    await page.goto("http://localhost:3000");

    // Await for initialization
    await page.evaluate(() => window.__ailoyLoaded);

    // Run OPFS put/get in browser context
    const testData = "Hello, OPFS!";
    const result = await page.evaluate(async (data) => {
      const encoded = new TextEncoder().encode(data);

      try {
        await window.ailoy.ailoy_filesystem_write("foo/bar", encoded);
        const result = await window.ailoy.ailoy_filesystem_read("foo/bar");
        await window.ailoy.ailoy_filesystem_remove("foo");
        return new TextDecoder().decode(result);
      } catch (err) {
        return `Error: ${err}`;
      }
    }, testData);
    expect(result).toBe(testData);
  });
});
