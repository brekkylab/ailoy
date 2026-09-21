// Fetching a workspace file's bytes, for the viewers that open a format rather than read
// characters.
//
// Over `invoke`, like everything else in this app. It was a custom URI scheme first, which
// streams and costs no copy, and it never worked: a scheme has to clear the CSP, satisfy
// CORS against the window's own origin, and be registered before the webview exists, and
// all three are only observable inside the packaged app. The failure looked like every
// other failure and left nothing in the log to tell them apart.
//
// The bridge is the transport this app already proves on every other call. `fs_read_bytes`
// answers with `tauri::ipc::Response`, so the buffer crosses as bytes rather than as a JSON
// array of numbers, and arrives here as an `ArrayBuffer`.

import { useEffect, useState } from "react";

import * as api from "@/api";

export type Bytes =
  | { state: "loading" }
  | { state: "ready"; bytes: ArrayBuffer }
  | { state: "failed" };

/**
 * The caller must key the component on `path`, so a different file arrives as a fresh
 * mount. That is what lets this start in `loading` and never go back to it: resetting the
 * state inside the effect would be a second render for every read, and the one case it
 * guards against — a path changing under a live instance — cannot happen when the
 * component is keyed.
 */
export function useBytes(path: string): Bytes {
  const [result, setResult] = useState<Bytes>({ state: "loading" });
  useEffect(() => {
    let live = true;
    void api.fsReadBytes(path).then(
      (bytes) => {
        if (live) setResult({ state: "ready", bytes });
      },
      (err: unknown) => {
        // The pane has room for one sentence and says it. The reason goes here, so a file
        // that will not open leaves something to read rather than only a red line.
        console.warn(`${path} could not be read`, err);
        if (live) setResult({ state: "failed" });
      },
    );
    return () => {
      live = false;
    };
  }, [path]);
  return result;
}
