// Which rows of a thread are new enough to fade in.
//
// Only what arrives while the reader watches. A conversation opened from the list is
// history, and a thread of forty turns fading in at once would be a page flickering, not a
// message arriving; so the first list a session shows is taken as already seen. After that
// a row is new the first time its key is on screen — a user's message, a tool call, the
// block a run starts streaming into.
//
// With one exception, and it is what makes this more than a set of keys: an answer the
// reader has already watched stream in is replaced, when it is stored, by a row with a key
// of its own. Fading that row in would blank an answer they were reading and bring it back,
// so a stored answer that takes over from streamed text arrives without one.
//
// What has been seen is recorded as a new state, never by mutating the old one — StrictMode
// renders twice, and a render that marked a key seen in place would leave the second one,
// which is the one kept, with nothing new in it. And a key that entered stays entering: the
// class only plays its animation when it is first applied, so keeping it is free, while
// taking it off on the next render would cut the fade off a frame in.

import { useState } from "react";

/** The fade a new row comes in with: a short rise and a fade from nothing. */
export const ENTER = "animate-in fade-in-0 slide-in-from-bottom-1 duration-300 ease-out";

export interface EntranceRow {
  key: string;
  /** A stored assistant answer — the kind of row that takes over from streamed text. */
  answer?: boolean;
}

interface State {
  session: string | null;
  primed: boolean;
  seen: ReadonlySet<string>;
  entering: ReadonlySet<string>;
  /** Streamed text has been on screen that no stored answer has taken over from yet. */
  streamed: boolean;
}

const fresh = (session: string | null): State => ({
  session,
  primed: false,
  seen: new Set(),
  entering: new Set(),
  streamed: false,
});

/**
 * What the rows on screen now change about what has been seen — the same state back when
 * they change nothing, which is what lets the caller skip the update. Pure, and exported so
 * the rules are tested without a webview.
 */
export function recordEntrance(s: State, ready: boolean, rows: EntranceRow[], streaming: boolean): State {
  if (!ready) return s;
  if (!s.primed) {
    return { ...s, primed: true, seen: new Set(rows.map((r) => r.key)), streamed: streaming };
  }
  let seen: Set<string> | null = null;
  let entering: Set<string> | null = null;
  let streamed = s.streamed;
  for (const r of rows) {
    if (s.seen.has(r.key) || seen?.has(r.key)) continue;
    seen ??= new Set(s.seen);
    seen.add(r.key);
    if (r.answer && streamed) streamed = false;
    else (entering ??= new Set(s.entering)).add(r.key);
  }
  if (streaming) streamed = true;
  if (!seen && streamed === s.streamed) return s;
  return { ...s, seen: seen ?? s.seen, entering: entering ?? s.entering, streamed };
}

/** Whether a row fades in, by the state as the last commit left it. */
export function isEntering(s: State, row: EntranceRow): boolean {
  return s.entering.has(row.key) || (s.primed && !s.seen.has(row.key) && !(row.answer && s.streamed));
}

/**
 * Whether a row should fade in, for the rows on screen now.
 *
 * `ready` is whether the session's stored list has arrived — the list that is history.
 * `streaming` is whether streamed text is on screen this render.
 */
export function useEntrance(
  session: string | null,
  ready: boolean,
  rows: EntranceRow[],
  streaming: boolean,
): (row: EntranceRow) => boolean {
  const [held, setHeld] = useState(() => fresh(session));
  // Another session is another history.
  const base = held.session === session ? held : fresh(session);
  // This render is judged by what was seen before it, and what it shows is recorded for the
  // next — React's own pattern for state that follows what a render was given: set while
  // rendering, and only when it changes, which is when a row is seen for the first time or
  // the stream starts or is taken over. A few times a run, not once per streamed token.
  const next = recordEntrance(base, ready, rows, streaming);
  if (next !== held) setHeld(next);
  return (row) => isEntering(base, row);
}
