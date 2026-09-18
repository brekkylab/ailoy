// Through `@/api`, the surface components import: importing `@tauri-apps/api/core` under
// node is fine (only *constructing* a `Channel` needs a webview), so this also pins that
// the re-export from `lib/errors` stays wired.
import { describe, expect, it } from "vitest";

import { kindOf, messageOf } from "@/api";

describe("messageOf", () => {
  it("reads the engine's message off a rejection payload", () => {
    expect(messageOf({ kind: "not_found", message: "not found: session s1" })).toBe("not found: session s1");
    // `Invalid` carries text the user reads verbatim.
    expect(messageOf({ kind: "invalid", message: "Enter a title" })).toBe("Enter a title");
  });

  it("falls back for anything else that can be thrown", () => {
    expect(messageOf("plain string")).toBe("plain string");
    expect(messageOf(new Error("boom"))).toBe("boom");
    expect(messageOf(null)).toBe("null");
    expect(messageOf(42)).toBe("42");
  });
});

describe("kindOf", () => {
  it("returns the tag to branch on", () => {
    expect(kindOf({ kind: "already_running", message: "session is already running" })).toBe("already_running");
    expect(kindOf({ kind: "console_unavailable", message: "console unavailable: no kernel" })).toBe("console_unavailable");
  });

  it("is null for a failure that did not come from the engine", () => {
    expect(kindOf(new Error("boom"))).toBeNull();
    expect(kindOf("not found")).toBeNull();
    expect(kindOf(null)).toBeNull();
    // A tag this build does not know is not one the UI may branch on.
    expect(kindOf({ kind: "from_a_newer_engine", message: "?" })).toBeNull();
  });
});
