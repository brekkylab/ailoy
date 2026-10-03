// Reporting a failure the window has already shown the user.
//
// A packaged webview's console cannot be opened, so `console.warn` alone reaches nobody
// outside a dev build. This also sends the line to the engine, which writes it to the log
// file — the one place a user can find it and hand it over.
//
// For failures, not for tracing: the pane says one sentence, and this records the reason
// behind it.

import * as api from "@/api";

/** What an unknown thrown value has to say for itself. */
function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

export function report(what: string, err?: unknown): void {
  const line = err === undefined ? what : `${what}: ${describe(err)}`;
  console.warn(line, err);
  // Fire and forget, and never throw: a failure to report a failure is not worth a second
  // one, and this is called from paths that are already handling something else.
  void api.clientLog("warn", line).catch(() => {});
}
