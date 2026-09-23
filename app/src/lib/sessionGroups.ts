import type { SessionSummary } from "@/types";

export interface SessionGroup {
  label: string;
  sessions: SessionSummary[];
}

const DAY = 24 * 60 * 60 * 1000;

function dayStart(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/** Groups sessions (newest first) under headed stretches of time. */
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
