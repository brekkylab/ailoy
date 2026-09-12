import { describe, expect, it } from "vitest";

import { S } from "@/strings";

describe("strings", () => {
  it("has an app name", () => {
    expect(S.appName).toBe("Ailoy");
  });
});
