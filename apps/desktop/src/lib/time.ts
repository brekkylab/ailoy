// "When was this last touched", in as few characters as a sidebar row can spare.

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
/** Past this, a count of days stops being easier to read than the date itself. */
const RELATIVE_LIMIT = 7 * DAY;

/** Local calendar date, zero-padded. `toISOString` would be UTC and could name yesterday. */
function isoDate(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/**
 * A timestamp as relative time.
 *
 * `now` is a parameter and not `Date.now()` so that a list of rows all measure against the
 * same instant — and so that this is testable without freezing the clock.
 *
 * A timestamp in the future reads as `just now`: the engine's clock and the webview's are
 * the same clock, so any lead is a rounding artefact or a machine whose time just changed,
 * and "in 3 minutes" for a session that already exists is worse than saying nothing new.
 */
export function formatRelativeTime(ms: number, now: number): string {
  if (!Number.isFinite(ms)) return "";
  const delta = now - ms;
  if (delta < MINUTE) return "just now";
  if (delta < HOUR) return `${Math.floor(delta / MINUTE)}m ago`;
  if (delta < DAY) return `${Math.floor(delta / HOUR)}h ago`;
  if (delta < RELATIVE_LIMIT) return `${Math.floor(delta / DAY)}d ago`;
  return isoDate(new Date(ms));
}

/**
 * When a message was written, as the time beside it says it: `15:04` today, `Sep 22, 15:04`
 * on another day this year, `Sep 22, 2025, 15:04` before that. A clock rather than "3h ago",
 * because a message's time is read to place it in the day, not to measure a gap.
 */
export function formatMessageTime(ms: number, now: number): string {
  if (!Number.isFinite(ms)) return "";
  const d = new Date(ms);
  const n = new Date(now);
  const clock = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  if (d.toDateString() === n.toDateString()) return clock;
  const date = d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(d.getFullYear() === n.getFullYear() ? {} : { year: "numeric" }),
  });
  return `${date}, ${clock}`;
}
