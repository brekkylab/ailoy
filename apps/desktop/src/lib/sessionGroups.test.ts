import { describe, expect, it } from "vitest";

import type { SessionSummary } from "@/types";

import { groupSessions } from "./sessionGroups";

const at = (y: number, m: number, d: number, h = 12) => new Date(y, m - 1, d, h).getTime();
const session = (id: string, updated_at: number): SessionSummary => ({
  id,
  title: id,
  model: "anthropic/m",
  created_at: updated_at,
  updated_at,
  running: false,
});

describe("the session list's headings", () => {
  // A Wednesday afternoon.
  const now = at(2026, 9, 23, 15);

  it("cuts the newest-first list into stretches of calendar days", () => {
    const groups = groupSessions(
      [
        session("a", at(2026, 9, 23, 9)),
        session("b", at(2026, 9, 22, 23)),
        session("c", at(2026, 9, 19)),
        session("d", at(2026, 9, 5)),
        session("e", at(2026, 7, 30)),
        session("f", at(2026, 7, 2)),
        session("g", at(2025, 12, 31)),
      ],
      now,
    );
    expect(groups.map((g) => [g.label, g.sessions.map((s) => s.id)])).toEqual([
      ["Today", ["a"]],
      ["Yesterday", ["b"]],
      ["Previous 7 days", ["c"]],
      ["Previous 30 days", ["d"]],
      ["July 2026", ["e", "f"]],
      ["December 2025", ["g"]],
    ]);
  });

  it("reads a day as a calendar day, not as 24 hours", () => {
    // 11pm the day before is yesterday's even at 1am.
    const early = at(2026, 9, 23, 1);
    expect(groupSessions([session("late", at(2026, 9, 22, 23))], early)[0].label).toBe("Yesterday");
  });

  it("counts a session from the future as today", () => {
    expect(groupSessions([session("x", at(2026, 9, 25))], now)[0].label).toBe("Today");
    expect(groupSessions([], now)).toEqual([]);
  });
});
