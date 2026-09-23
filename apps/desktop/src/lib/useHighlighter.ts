// The shared highlighter, as a hook.
//
// Its own module because two components want the same one: the thread's fences and the
// file panel's source viewer. Building a second highlighter would mean a second copy of
// every grammar, and the point of `lib/highlighter` is that there is one.

import { useEffect, useState } from "react";

import { getHighlighter, type Highlighter } from "@/lib/highlighter";

/**
 * The built highlighter, once it exists.
 *
 * Module-level rather than per-component: every fence in every bubble asks for it, and a
 * mount after the first must not paint a plain block while a promise that has already
 * resolved is awaited again. A failure to build it is not worth a message — everything
 * keeps rendering as plain text, which is what an unknown language does anyway.
 */
let ready: Highlighter | null = null;

export function useHighlighter(): Highlighter | null {
  const [highlighter, setHighlighter] = useState(ready);
  useEffect(() => {
    if (highlighter) return;
    let live = true;
    void getHighlighter().then(
      (h) => {
        ready = h;
        if (live) setHighlighter(h);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  }, [highlighter]);
  return highlighter;
}
