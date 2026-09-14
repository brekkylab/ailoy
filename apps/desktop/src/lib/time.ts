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
 * A timestamp as Korean relative time.
 *
 * `now` is a parameter and not `Date.now()` so that a list of rows all measure against the
 * same instant — and so that this is testable without freezing the clock.
 *
 * A timestamp in the future reads as `방금`: the engine's clock and the webview's are the
 * same clock, so any lead is a rounding artefact or a machine whose time just changed, and
 * "in 3 minutes" for a session that already exists is worse than saying nothing new.
 */
export function formatRelativeTime(ms: number, now: number): string {
  if (!Number.isFinite(ms)) return "";
  const delta = now - ms;
  if (delta < MINUTE) return "방금";
  if (delta < HOUR) return `${Math.floor(delta / MINUTE)}분 전`;
  if (delta < DAY) return `${Math.floor(delta / HOUR)}시간 전`;
  if (delta < RELATIVE_LIMIT) return `${Math.floor(delta / DAY)}일 전`;
  return isoDate(new Date(ms));
}
