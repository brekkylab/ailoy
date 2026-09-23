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
 * How long ago a message was written, in words: `just now`, `1 minute ago`, `5 hours ago`,
 * `3 days ago`, `2 months ago`, `1 year ago`. Spelled out rather than the sidebar's `5m ago`:
 * under a message there is the room, and it is read as a sentence. The exact time is the
 * element's tooltip. A time ahead of `now` is `just now`, for the reason `formatRelativeTime`
 * gives.
 */
export function formatElapsed(ms: number, now: number): string {
  if (!Number.isFinite(ms)) return "";
  const delta = now - ms;
  const count = (n: number, unit: string) => `${n} ${unit}${n === 1 ? "" : "s"} ago`;
  if (delta < MINUTE) return "just now";
  if (delta < HOUR) return count(Math.floor(delta / MINUTE), "minute");
  if (delta < DAY) return count(Math.floor(delta / HOUR), "hour");
  if (delta < 30 * DAY) return count(Math.floor(delta / DAY), "day");
  if (delta < 365 * DAY) return count(Math.floor(delta / (30 * DAY)), "month");
  return count(Math.floor(delta / (365 * DAY)), "year");
}
