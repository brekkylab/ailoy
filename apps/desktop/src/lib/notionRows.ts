// What a Notion row looked like last time, so the tree does not start blank.
//
// A row needs two things the listing above it does not carry: the page's icon, and whether
// it holds any sub-pages. Both live in the page's own json, so the tree reads one per
// visible row — a page render on cortex's side, which is a retrieve and, cold, a walk of
// the page's whole block tree. cortex keeps those renders between runs, so the *second*
// time a window asks it is cheap. It is still a request, and there are as many of them as
// there are rows on screen, which is the second or two of icons and chevrons arriving one
// by one after a restart.
//
// So the answers are remembered here too, and a restored row is drawn from them at once:
// the read still happens, and what it finds replaces this. Being wrong costs an icon that
// corrects itself a moment later, which is what makes this safe to keep for a long time —
// unlike a file's contents, nothing is decided from it.

/** How many rows are remembered. A page tree is browsed downwards; this is generous for one. */
const LIMIT = 1000;

const KEY = "ailoy.notionRows";

/** What a row draws with, apart from what its path says. */
export interface RowMemo {
  /** The page's emoji, or `null` for one that has none. */
  icon: string | null;
  /** Whether the page holds nothing — no sub-pages, or no rows for a database. */
  leaf: boolean;
  /** The page's own title, spaces and all, or `null` when the json did not carry one. */
  title: string | null;
}

/** Stored compactly: this is one blob of every row a reader has seen. */
type Stored = Record<string, [icon: string | null, leaf: boolean, title?: string | null]>;

function read(): Stored {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Stored) : {};
  } catch {
    // Storage may be unavailable, hold something that is not ours, or hold a shape written
    // by an older build. All three read as "nothing remembered", which is never wrong: the
    // row simply waits for its read, the way it did before any of this.
    return {};
  }
}

export function recalledRow(path: string): RowMemo | null {
  const entry = read()[path];
  if (!Array.isArray(entry) || entry.length < 2) return null;
  const [icon, leaf, title] = entry;
  if ((icon !== null && typeof icon !== "string") || typeof leaf !== "boolean") return null;
  // A blob written before rows remembered their titles has two fields, not three. It is
  // still a good answer to the two it has, and the title fills in with the next read.
  return { icon, leaf, title: typeof title === "string" ? title : null };
}

/**
 * Remember what `path`'s json said.
 *
 * Capped by dropping the oldest writes: object key order is insertion order, and a reader
 * moves through a tree, so the oldest keys are the rows furthest behind them.
 */
export function rememberRow(path: string, memo: RowMemo) {
  try {
    const stored = read();
    // Delete first, so re-seeing a row moves it to the end rather than leaving it where it
    // was — otherwise the cap would drop the rows a reader keeps coming back to.
    delete stored[path];
    stored[path] = [memo.icon, memo.leaf, memo.title];
    const keys = Object.keys(stored);
    for (const old of keys.slice(0, Math.max(0, keys.length - LIMIT))) delete stored[old];
    localStorage.setItem(KEY, JSON.stringify(stored));
  } catch {
    /* storage may be unavailable, or full; the row is already drawn either way */
  }
}
