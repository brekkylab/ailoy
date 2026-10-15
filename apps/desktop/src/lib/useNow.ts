// One clock for everything on screen that says how long ago.
//
// A row that reads "3 minutes ago" goes stale while nothing about the data changes, so it
// needs a tick of its own — and a thread has one per message. They all read the same
// ticker here: one interval, running only while something is listening.

import { useSyncExternalStore } from "react";

const TICK_MS = 30_000;

let now = Date.now();
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | null = null;

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (timer === null) {
    now = Date.now();
    timer = setInterval(() => {
      now = Date.now();
      for (const l of listeners) l();
    }, TICK_MS);
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };
}

/** The current time, advanced every 30 seconds for as long as the component is mounted. */
export function useNow(): number {
  return useSyncExternalStore(subscribe, () => now);
}
