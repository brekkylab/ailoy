import { describe, expect, it } from "vitest";

import { sessionTitle } from "@/lib/sessionTitle";
import type { SessionSummary } from "@/types";

const row = (id: string, title: string): SessionSummary => ({
  id,
  title,
  model: "anthropic/claude-opus-5",
  created_at: 0,
  updated_at: 0,
  running: false,
});

const list = [row("a", "Revenue review"), row("b", "New chat")];

describe("sessionTitle", () => {
  it("names the open session", () => {
    expect(sessionTitle("a", list)).toBe("Revenue review");
  });

  it("names a session that still carries the default title", () => {
    // There is no auto-titling, so this is what most sessions read as. Blanking it here
    // would leave the bar empty for the whole life of most windows.
    expect(sessionTitle("b", list)).toBe("New chat");
  });

  it("is blank when no session is open", () => {
    expect(sessionTitle(null, list)).toBeNull();
  });

  it("is blank until the list arrives", () => {
    expect(sessionTitle("a", undefined)).toBeNull();
  });

  it("is blank for a session the list no longer has", () => {
    // The id outlives the list across a delete: the bar waits rather than naming the row
    // that happens to be first.
    expect(sessionTitle("gone", list)).toBeNull();
  });

  it("is blank for a session stored with an empty title", () => {
    expect(sessionTitle("c", [...list, row("c", "")])).toBeNull();
  });
});
