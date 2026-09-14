// `attachRun` itself cannot be exercised here — `new Channel()` needs a webview — so the
// rule it turns on is tested where it lives.
import { describe, expect, it } from "vitest";

import { shouldAttach } from "./events";

describe("shouldAttach", () => {
  it("attaches a session that has no live channel", () => {
    expect(shouldAttach(new Set<string>(), "s1")).toBe(true);
    expect(shouldAttach(new Set(["s2"]), "s1")).toBe(true);
  });

  it("refuses a second attach while one channel is already feeding the session", () => {
    expect(shouldAttach(new Set(["s1"]), "s1")).toBe(false);
    expect(shouldAttach(new Map([["s1", {}]]), "s1")).toBe(false);
  });
});
