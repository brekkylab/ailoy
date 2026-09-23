// The run event channel, wired to the run store.
//
// One channel per run: Rust pumps a run's events into it and closes it after the terminal
// event (`done`, `cancelled`, `error`), so a channel is never reused across runs. Callers
// get a promise and read the run's live state from `useRunStore`.
//
// The module also remembers which sessions currently hold a live channel, because
// `run_attach` is called from a mount effect and React mounts a component more than once
// (StrictMode in dev, a remount when the session list settles). Attaching twice would
// hand the engine two channels for one run: every event would be applied to the store
// twice, and `run_attach`'s replay would paint the buffered text a second time.

import { Channel } from "@tauri-apps/api/core";

import * as api from "./api";
import { useRunStore } from "./store/runs";
import type { RunEvent } from "./types";

/** Sessions whose run is already feeding this module a channel. */
const liveChannels = new Map<string, Channel<RunEvent>>();

/**
 * Whether `attachRun` should actually reach into the engine.
 *
 * Split out and exported because the decision is the whole of the dedupe rule and
 * `new Channel()` needs a webview, so `attachRun` itself cannot run under vitest.
 * Anything with a `has` answers here — the live registry is a `Map`, a test may pass a
 * `Set`.
 */
export function shouldAttach(live: { has(id: string): boolean }, sessionId: string): boolean {
  return !live.has(sessionId);
}

const TERMINAL: ReadonlySet<RunEvent["type"]> = new Set<RunEvent["type"]>(["done", "cancelled", "error"]);

function channelFor(sessionId: string): Channel<RunEvent> {
  const ch = new Channel<RunEvent>();
  ch.onmessage = (ev) => {
    // Deregister *before* applying: a component that reacts to the terminal status
    // synchronously (a remount, a retry) must already see the session as attachable.
    // The identity check keeps a late event from a superseded channel from evicting the
    // current one.
    if (TERMINAL.has(ev.type) && liveChannels.get(sessionId) === ch) liveChannels.delete(sessionId);
    useRunStore.getState().apply(sessionId, ev);
  };
  // A fresh channel replaces whatever was registered: `run_start` is the authoritative
  // new run, and a stale entry can only belong to a run that ended without one of its
  // terminal events reaching us.
  liveChannels.set(sessionId, ch);
  return ch;
}

/** Drop a registration that turned out to carry no run, so a later attach may retry. */
function forget(sessionId: string, ch: Channel<RunEvent>) {
  if (liveChannels.get(sessionId) === ch) liveChannels.delete(sessionId);
}

export async function startRun(sessionId: string, text: string): Promise<string> {
  const ch = channelFor(sessionId);
  try {
    return await api.runStart(sessionId, text, ch);
  } catch (err) {
    forget(sessionId, ch);
    throw err;
  }
}

/**
 * After a reload: re-subscribe if a run is still going. Resolves to the run id, or to
 * `null` both when nothing is running and when this session is already subscribed.
 */
export async function attachRun(sessionId: string): Promise<string | null> {
  if (!shouldAttach(liveChannels, sessionId)) return null;
  const ch = channelFor(sessionId);
  try {
    const runId = await api.runAttach(sessionId, ch);
    if (runId === null) forget(sessionId, ch);
    return runId;
  } catch (err) {
    forget(sessionId, ch);
    throw err;
  }
}

export const cancelRun = (sessionId: string) => api.runCancel(sessionId);
