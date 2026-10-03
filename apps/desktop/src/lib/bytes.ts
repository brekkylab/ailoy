// The file bytes a viewer opened, kept for as long as they fit.
//
// A PDF or a deck is read whole, over the bridge, from a store that may be across a network:
// the same 5 MiB file read twice showed as 597ms and 529ms in one session's timings. So the
// read is a query like any other — cached by path, never stale on its own, and invalidated
// where every other file query already is (a finished run, a source connected or removed).
//
// What a query cache cannot do is stop at a size, and these are the only entries in it with a
// size worth minding. `trimBytes` is that bound: after a read lands, the oldest entries go
// until what is kept fits the budget.
//
// Newest-first and not least-recently-used, for the reason the stores below say it: a reader
// opens files in the order they are working through them, and keeping a use counter in step
// with every render would cost more here than it saves. The file on screen is the newest read,
// so it is the one thing never dropped — even when it alone is over budget.

import type { QueryClient } from "@tanstack/react-query";

/** The query key prefix. One place, because two files invalidate it. */
export const BYTES_KEY = "bytes";

/**
 * How much of the window's memory the read files may hold.
 *
 * The engine refuses a file above 64 MiB, so one entry can be that large; this holds roughly
 * a dozen of the sizes that actually turn up (a scanned PDF, a deck of photographs) and a
 * single outsized one on its own.
 */
export const BYTES_BUDGET = 64 << 20;

/** How long an unused entry survives before the query cache drops it anyway. */
export const BYTES_GC_MS = 30 * 60_000;

function sizeOf(data: unknown): number {
  return data instanceof ArrayBuffer ? data.byteLength : 0;
}

/** Drop read files, oldest first, until what is left fits `budget`. */
export function trimBytes(qc: QueryClient, budget = BYTES_BUDGET) {
  const held = qc
    .getQueryCache()
    .findAll({ queryKey: [BYTES_KEY] })
    .map((q) => ({ key: q.queryKey, at: q.state.dataUpdatedAt, size: sizeOf(q.state.data) }))
    .filter((e) => e.size > 0)
    .sort((a, b) => b.at - a.at);

  let kept = 0;
  for (const [i, entry] of held.entries()) {
    kept += entry.size;
    // `i > 0`: the newest read is what the viewer on screen is drawing from, and dropping it
    // would be a re-read of the file the reader is looking at.
    if (i > 0 && kept > budget) qc.removeQueries({ queryKey: entry.key, exact: true });
  }
}
