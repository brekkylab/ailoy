import type { hello } from "../src";
import { test, expect } from "@playwright/test";

test("ailoy", async ({ page }) => {
  await page.goto("/");

  page.on("console", (msg) => {
    console.log(msg);
  });

  const result = await page.evaluate(async () => {
    /// @ts-ignore
    const hello: Runtime = new window.Ailoy.hello();
    hello();
  });
  expect(result, "Subscribe send failed");
});
