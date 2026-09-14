// What a run looks like while it is happening.
//
// `applyRunEvent` is a pure reducer over the engine's event stream — every rule about
// what the thread paints lives there and is tested without a webview. The zustand store
// is the thin part: one `LiveRun` per session id, so switching sessions never loses a
// run in flight.

import { useEffect, useRef } from "react";
import { create } from "zustand";

import type { Message, RateLimitInfo, RunEvent, TokenUsage } from "@/types";

export type ToolStatus = "running" | "done" | "error" | "interrupted";
export interface ToolCallState { id: string; name: string; arguments: unknown; status: ToolStatus; result?: unknown; startedAt: number; finishedAt?: number }

export interface LiveRun {
  runId: string | null;
  /**
   * Terminal states are final for this run: the engine drops the run from its active map
   * and closes the channel after `done`/`cancelled`/`error`, so nothing else arrives
   * until the next `started`.
   */
  status: "idle" | "running" | "done" | "cancelled" | "error";
  text: string;
  thinking: string;
  toolCalls: Record<string, ToolCallState>;
  toolOrder: string[];
  /** Merged totals of the last completed model message — replaced, not summed. */
  usage?: TokenUsage | null;
  rateLimit?: RateLimitInfo | null;
  contextUsed?: number | null;
  contextLimit?: number | null;
  /** `status`/`retryable` are filled only for `kind: "model"`. */
  error?: { kind: string; message: string; status: number | null; retryable: boolean };
  /**
   * How many messages the engine has persisted for this session since this store started
   * watching. A counter, not a flag: a boolean latch cleared after the thread's refetch
   * swallows any `message` that arrives *during* that refetch — the clear erases a signal
   * raised for a message the query never saw. The thread acks the version it actually
   * fetched, so a message landing mid-flight leaves `messagesVersion > messagesAcked` and
   * starts another round.
   */
  messagesVersion: number;
  /** The highest `messagesVersion` the thread has refetched. Never moves backwards. */
  messagesAcked: number;
}

export const emptyRun = (): LiveRun => ({ runId: null, status: "idle", text: "", thinking: "", toolCalls: {}, toolOrder: [], messagesVersion: 0, messagesAcked: 0 });

function toolResultValue(msg: Message): unknown {
  const first = msg.contents[0];
  if (!first) return null;
  if (first.type === "value") return first.value;
  if (first.type === "text") return first.text;
  return first;
}

export function applyRunEvent(s: LiveRun, ev: RunEvent): LiveRun {
  switch (ev.type) {
    // Also how a re-attach begins: `run_attach` synthesizes a `started` and replays the
    // buffered text as one `text_delta`, so resetting here is what keeps a reload from
    // painting the previous mount's text twice.
    // The message counters survive the reset: they describe the session's stored list,
    // not this run, and rewinding them to zero while the thread still holds a higher ack
    // would leave the pair permanently mismatched.
    case "started":
      return { ...emptyRun(), runId: ev.run_id, status: "running", messagesVersion: s.messagesVersion, messagesAcked: s.messagesAcked };
    case "text_delta":
      return { ...s, text: s.text + ev.text };
    case "thinking_delta":
      return { ...s, thinking: s.thinking + ev.text };
    case "tool_call_started":
      return {
        ...s,
        toolCalls: { ...s.toolCalls, [ev.id]: { id: ev.id, name: ev.name, arguments: ev.arguments, status: "running", startedAt: Date.now() } },
        toolOrder: s.toolOrder.includes(ev.id) ? s.toolOrder : [...s.toolOrder, ev.id],
      };
    case "message": {
      const m = ev.message;
      if (m.role === "tool" && m.id && s.toolCalls[m.id]) {
        const value = toolResultValue(m);
        const isError = typeof value === "object" && value !== null && "error" in (value as Record<string, unknown>);
        return { ...s, messagesVersion: s.messagesVersion + 1, toolCalls: { ...s.toolCalls, [m.id]: { ...s.toolCalls[m.id], status: isError ? "error" : "done", result: value, finishedAt: Date.now() } } };
      }
      if (m.role === "assistant" && ev.depth === 0) {
        return { ...s, text: "", thinking: "", messagesVersion: s.messagesVersion + 1 };
      }
      return { ...s, messagesVersion: s.messagesVersion + 1 };
    }
    // Merged totals for one completed model message: a present field replaces, a null
    // field keeps what was there.
    case "usage":
      return { ...s, usage: ev.usage ?? s.usage, rateLimit: ev.rate_limit ?? s.rateLimit, contextUsed: ev.context_used ?? s.contextUsed, contextLimit: ev.context_limit ?? s.contextLimit };
    case "awaiting_approval":
      return s;
    case "done":
      return { ...s, status: "done" };
    case "cancelled":
      return { ...s, status: "cancelled", toolCalls: interruptRunning(s.toolCalls) };
    case "error":
      return { ...s, status: "error", error: { kind: ev.kind, message: ev.message, status: ev.status, retryable: ev.retryable }, toolCalls: interruptRunning(s.toolCalls) };
  }
}

function interruptRunning(calls: Record<string, ToolCallState>): Record<string, ToolCallState> {
  const out: Record<string, ToolCallState> = {};
  for (const [id, c] of Object.entries(calls)) out[id] = c.status === "running" ? { ...c, status: "interrupted", finishedAt: Date.now() } : c;
  return out;
}

interface RunStore {
  runs: Record<string, LiveRun>;
  apply: (sessionId: string, ev: RunEvent) => void;
  ackMessages: (sessionId: string, version: number) => void;
  reset: (sessionId: string) => void;
}

export const useRunStore = create<RunStore>((set) => ({
  runs: {},
  apply: (sessionId, ev) => set((st) => ({ runs: { ...st.runs, [sessionId]: applyRunEvent(st.runs[sessionId] ?? emptyRun(), ev) } })),
  // Monotonic: two refetches in flight can resolve in either order, and the older one
  // must not pull the ack back under the newer one.
  ackMessages: (sessionId, version) =>
    set((st) => {
      const run = st.runs[sessionId];
      // Clamped to the current version: an ack from a refetch that started before a
      // `reset` must not leave `acked` above `version`, or the next message would land
      // exactly on the ack and never be fetched.
      const next = Math.min(version, run?.messagesVersion ?? version);
      if (!run || run.messagesAcked >= next) return st;
      return { runs: { ...st.runs, [sessionId]: { ...run, messagesAcked: next } } };
    }),
  reset: (sessionId) => set((st) => ({ runs: { ...st.runs, [sessionId]: emptyRun() } })),
}));

/**
 * One shared "nothing here yet" value. A selector that built a fresh object per call
 * would hand React a new snapshot on every render and never settle.
 *
 * Frozen all the way down: the collections are shared by every idle reader, so a
 * component that pushed onto `toolOrder` or assigned into `toolCalls` would poison every
 * other session's idle state. Freezing turns that into a throw at the write.
 */
const IDLE: LiveRun = (() => {
  const idle = emptyRun();
  Object.freeze(idle.toolCalls);
  Object.freeze(idle.toolOrder);
  return Object.freeze(idle);
})();

export const selectRun = (sessionId: string | null) => (st: RunStore) => (sessionId ? st.runs[sessionId] ?? IDLE : IDLE);

/**
 * A run is over. The engine drops it from its active map and closes the channel after
 * one of these, so the transition happens exactly once per run.
 */
export const isTerminal = (s: LiveRun["status"]) => s === "done" || s === "cancelled" || s === "error";

/**
 * Calls `onTerminal` once, each time this session's run *crosses into* a terminal state.
 *
 * The transition and not the state: `status` stays `done` until the next run starts, so a
 * plain `if (isTerminal(status))` in an effect would fire again on every unrelated
 * re-render. The remembered status is keyed by session id so that switching sessions does
 * not read the other one's `done` as this one's ending.
 *
 * Everything a run writes on its way out — the messages, the session's usage row and
 * `updated_at`, and whatever the agent's tools did to the workspace — lands at this
 * moment and at no other, which is why one hook serves all of it.
 */
export function useRunTerminal(sessionId: string | null, onTerminal: () => void): void {
  const { status } = useRunStore(selectRun(sessionId));
  // Read through a ref: the caller passes a fresh closure every render, and depending on
  // it would re-run the effect below constantly — re-arming `seen` with the current status
  // and so swallowing the very transition this exists to catch. Restocked in an effect
  // rather than during render, and declared first so that it has run by the time the
  // effect below reads it.
  const cb = useRef(onTerminal);
  useEffect(() => {
    cb.current = onTerminal;
  });
  const seen = useRef<{ id: string | null; status: LiveRun["status"] } | null>(null);
  useEffect(() => {
    const was = seen.current?.id === sessionId ? seen.current.status : null;
    seen.current = { id: sessionId, status };
    if (was == null || was === status || !isTerminal(status)) return;
    cb.current();
  }, [sessionId, status]);
}
