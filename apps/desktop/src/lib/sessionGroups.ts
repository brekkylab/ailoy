// The session list, cut into the stretches of time the chat apps cut theirs into.
//
// A flat list with a relative time on every row made the reader do the grouping, row by
// row; headed stretches do it once. Local calendar days, not 24-hour windows, because
// "Yesterday" means the day before today to whoever reads it — a conversation from 11pm is
// yesterday's at 1am.

import type { SessionSummary } from "@/types";

export interface SessionGroup {
  label: string;
  sessions: SessionSummary[];
}

const DAY = 24 * 60 * 60 * 1000;

/** Local midnight at the start of the day `ms` falls in. */
function dayStart(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/**
 * The sessions under their headings, in the order they arrive — `session_list` is newest
 * first, so each group is too, and so are the groups.
 *
 * Today, Yesterday, the rest of the last 7 days, the rest of the last 30, then a month at a
 * time ("September 2026"). A session from the future — a clock that moved back — counts as
 * today rather than as a heading of its own.
 */
export function groupSessions(sessions: SessionSummary[], now: number): SessionGroup[] {
  const today = dayStart(now);
  const labelOf = (ms: number): string => {
    const age = Math.round((today - dayStart(ms)) / DAY);
    if (age <= 0) return "Today";
    if (age === 1) return "Yesterday";
    if (age < 7) return "Previous 7 days";
    if (age < 30) return "Previous 30 days";
    return new Date(ms).toLocaleDateString("en-US", { month: "long", year: "numeric" });
  };
  const groups: SessionGroup[] = [];
  for (const s of sessions) {
    const label = labelOf(s.updated_at);
    const last = groups[groups.length - 1];
    if (last?.label === label) last.sessions.push(s);
    else groups.push({ label, sessions: [s] });
  }
  return groups;
}
