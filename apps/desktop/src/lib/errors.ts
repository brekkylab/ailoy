// Reading a rejected command.
//
// Every engine command rejects with `{ kind, message }` (see `EngineError` in
// `apps/desktop/core/src/error.rs`), but a rejection can also be a plain `Error` thrown
// before the IPC boundary — a missing command, a serialization failure — so both
// helpers are total: they answer for anything caught.
//
// Kept out of `api.ts` so tests may exercise them without pulling in
// `@tauri-apps/api`, which needs a webview. `api.ts` re-exports both.

import type { EngineErrorKind } from "@/types";

const KINDS: readonly EngineErrorKind[] = [
  "not_found",
  "already_running",
  "invalid",
  "console_unavailable",
  "workspace",
  "storage",
  "io",
  "other",
];

/** The text to show the user. The engine already wrote it in their language. */
export function messageOf(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  if (typeof err === "object" && err !== null) {
    const message = (err as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return String(err);
}

/**
 * The tag to branch on, or `null` when the failure did not come from the engine (or
 * carries a tag this build does not know).
 */
export function kindOf(err: unknown): EngineErrorKind | null {
  if (typeof err !== "object" || err === null) return null;
  const kind = (err as { kind?: unknown }).kind;
  return typeof kind === "string" && (KINDS as readonly string[]).includes(kind)
    ? (kind as EngineErrorKind)
    : null;
}
