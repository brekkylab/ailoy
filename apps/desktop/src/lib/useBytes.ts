// Fetching a workspace file's bytes, for the two viewers that have to open a container
// themselves rather than hand a URL to something that already knows how.
//
// Through the same scheme the `<img>` and `<object>` use, so there is one path from the
// engine to the window and one place where a failure is reported.

import { useEffect, useState } from "react";

import { wsfileUrl } from "@/lib/wsfile";

export type Bytes =
  | { state: "loading" }
  | { state: "ready"; bytes: ArrayBuffer }
  | { state: "failed" };

/**
 * The caller must key the component on `path`, so a different file arrives as a fresh
 * mount. That is what lets this start in `loading` and never go back to it: resetting the
 * state inside the effect would be a second render for every fetch, and the one case it
 * guards against — a path changing under a live instance — cannot happen when the
 * component is keyed.
 */
export function useBytes(path: string): Bytes {
  const [result, setResult] = useState<Bytes>({ state: "loading" });
  useEffect(() => {
    let live = true;
    // `AbortController` so switching files mid-download stops the old one rather than
    // leaving it to finish into a component that is gone.
    const abort = new AbortController();
    fetch(wsfileUrl(path), { signal: abort.signal })
      .then(async (r) => {
        // The engine answers a refusal with its own message as the body. The pane has room
        // for one sentence and says it; this is where the rest of it goes, so a file that
        // will not open leaves something to read rather than only a red line.
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        return r.arrayBuffer();
      })
      .then((bytes) => {
        if (live) setResult({ state: "ready", bytes });
      })
      .catch((err: unknown) => {
        if (abort.signal.aborted) return;
        console.warn(`${path} could not be fetched`, err);
        if (live) setResult({ state: "failed" });
      });
    return () => {
      live = false;
      abort.abort();
    };
  }, [path]);
  return result;
}
