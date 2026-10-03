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
//
// A query and not a bare effect, so that reopening a file the reader has already opened costs
// nothing: the same file used to be read again on every mount, which for a file on a remote
// store is the whole round trip. See `lib/bytes` for what bounds that cache.
//
// **The buffer here is shared.** Every viewer holding this path is handed the same
// `ArrayBuffer`, and the cache keeps it after they unmount, so a caller that hands it to
// something which *transfers* it — a worker, `postMessage` — must pass a copy (`bytes.slice(0)`,
// which `PdfViewer` and `PptxViewer` already do). Transferring the original detaches it, and
// it would stay detached in the cache for every reader after.

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import * as api from "@/api";
import { BYTES_KEY, trimBytes } from "@/lib/bytes";
import { report } from "@/lib/report";

export type Bytes =
  | { state: "loading" }
  | { state: "ready"; bytes: ArrayBuffer }
  | { state: "failed" };

export function useBytes(path: string): Bytes {
  const qc = useQueryClient();
  const file = useQuery({
    queryKey: [BYTES_KEY, path],
    // Never stale on its own: a file's bytes change when something changes them, and the
    // events that do — a run finishing, a source connected or removed — invalidate this key
    // along with the rest of the file queries.
    staleTime: Infinity,
    queryFn: () => api.fsReadBytes(path),
  });

  // After a read lands, not during render: `trimBytes` writes to the same cache this is
  // reading from.
  const landed = file.dataUpdatedAt;
  useEffect(() => {
    if (landed) trimBytes(qc);
  }, [qc, landed]);

  // The pane has room for one sentence and says it. The reason goes to the log, so a file
  // that will not open leaves something to read rather than only a red line.
  const error = file.error;
  useEffect(() => {
    if (error) report(`${path} could not be read`, error);
  }, [path, error]);

  // One object per state, and not one per render.
  //
  // Every viewer keys an effect on this value — `[bytes, path]` — and opening a document sets
  // state, so a value that changed identity on each render would open the document, re-render,
  // tear it down, and open it again, forever. `PdfViewer` shows that as a blank pane: its
  // cleanup destroys the pdf.js task and terminates the worker, so the document is destroyed
  // the moment it arrives. A fresh object literal here did exactly that; the effect it
  // replaced held its value in `useState`, which is stable by construction.
  const data = file.data;
  return useMemo<Bytes>(
    () =>
      data ? { state: "ready", bytes: data } : error ? { state: "failed" } : { state: "loading" },
    [data, error],
  );
}
