// The run event channel, wired to the run store.
//
// One channel per call: Rust pumps a run's events into it and closes it after the
// terminal event (`done`, `cancelled`, `error`), so a channel is never reused across
// runs. Callers get a promise and read the run's live state from `useRunStore`.

import { Channel } from "@tauri-apps/api/core";

import * as api from "./api";
import { useRunStore } from "./store/runs";
import type { RunEvent } from "./types";

function channelFor(sessionId: string): Channel<RunEvent> {
  const ch = new Channel<RunEvent>();
  ch.onmessage = (ev) => useRunStore.getState().apply(sessionId, ev);
  return ch;
}

export async function startRun(sessionId: string, text: string): Promise<string> {
  return api.runStart(sessionId, text, channelFor(sessionId));
}

/** After a reload: re-subscribe if a run is still going. Resolves to the run id or null. */
export async function attachRun(sessionId: string): Promise<string | null> {
  return api.runAttach(sessionId, channelFor(sessionId));
}

export const cancelRun = (sessionId: string) => api.runCancel(sessionId);
