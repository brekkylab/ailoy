import { describe, expect, it } from "vitest";

import { applyRunEvent, emptyRun, selectRun, useRunStore } from "./runs";
import type { LiveRun } from "./runs";
import type { Message, RunEvent } from "@/types";

const asst = (text: string, toolCalls?: Message["tool_calls"]): Message => ({ role: "assistant", contents: [{ type: "text", text }], tool_calls: toolCalls });
const tool = (id: string, value: unknown): Message => ({ role: "tool", id, contents: [{ type: "value", value }] });

function run(events: RunEvent[]) {
  return events.reduce(applyRunEvent, emptyRun());
}

function runFrom(state: LiveRun, events: RunEvent[]) {
  return events.reduce(applyRunEvent, state);
}

describe("applyRunEvent", () => {
  it("accumulates text and thinking while running", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "thinking_delta", text: "hmm " },
      { type: "text_delta", text: "Hel" },
      { type: "text_delta", text: "lo" },
    ]);
    expect(s.status).toBe("running");
    expect(s.runId).toBe("r1");
    expect(s.text).toBe("Hello");
    expect(s.thinking).toBe("hmm ");
  });

  it("clears live text when the assistant message is persisted and flags a refetch", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "Hello" },
      { type: "message", seq: 2, depth: 0, source_agent: null, message: asst("Hello"), usage: null },
    ]);
    expect(s.text).toBe("");
    expect(s.messagesDirty).toBe(true);
  });

  it("tracks tool calls from started to done with the tool result", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "ls" } },
      { type: "message", seq: 3, depth: 0, source_agent: null, message: tool("c1", { stdout: "a\n" }), usage: null },
    ]);
    expect(s.toolOrder).toEqual(["c1"]);
    expect(s.toolCalls.c1.status).toBe("done");
    expect(s.toolCalls.c1.result).toEqual({ stdout: "a\n" });
  });

  it("marks running tools interrupted on cancel and keeps text", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "partial" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: {} },
      { type: "cancelled" },
    ]);
    expect(s.status).toBe("cancelled");
    expect(s.toolCalls.c1.status).toBe("interrupted");
    expect(s.text).toBe("partial");
  });

  it("records usage and errors", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "usage", usage: { input_tokens: 10, output_tokens: 2 }, rate_limit: { requests: { limit: 100, remaining: 99, reset_at_ms: null } }, context_used: 10, context_limit: 1000 },
      { type: "error", kind: "model", message: "401", status: 401, retryable: false },
    ]);
    expect(s.status).toBe("error");
    expect(s.error).toEqual({ kind: "model", message: "401", status: 401, retryable: false });
    expect(s.contextUsed).toBe(10);
    expect(s.rateLimit?.requests?.remaining).toBe(99);
  });

  it("keeps the previous accounting for a usage field that arrives null", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "usage", usage: { input_tokens: 10, output_tokens: 2 }, rate_limit: null, context_used: 10, context_limit: 1000 },
      { type: "usage", usage: { input_tokens: 30, output_tokens: 4 }, rate_limit: null, context_used: 30, context_limit: null },
    ]);
    // Merged totals for one model message: a present field replaces outright.
    expect(s.usage).toEqual({ input_tokens: 30, output_tokens: 4 });
    expect(s.contextUsed).toBe(30);
    // A null field says "nothing reported this turn", not "it is gone".
    expect(s.contextLimit).toBe(1000);
  });

  it("a new start resets the previous run's live state", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "old" },
      { type: "done" },
      { type: "started", run_id: "r2" },
    ]);
    expect(s.runId).toBe("r2");
    expect(s.text).toBe("");
    expect(s.toolOrder).toEqual([]);
  });

  it("re-attach replays the partial text into a fresh bubble", () => {
    // What a reload looks like: state left over from before it, then `run_attach`'s
    // synthesized `started`, the whole buffer as one `text_delta`, then live events.
    const stale = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "before-reload" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: {} },
    ]);
    const s = runFrom(stale, [
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "buffered-so-far" },
      { type: "text_delta", text: "-then-live" },
    ]);
    expect(s.status).toBe("running");
    expect(s.runId).toBe("r1");
    expect(s.text).toBe("buffered-so-far-then-live");
    expect(s.toolOrder).toEqual([]);
  });
});

describe("useRunStore", () => {
  it("keeps one run per session and clears the refetch flag", () => {
    const st = () => useRunStore.getState();
    st().apply("s1", { type: "started", run_id: "r1" });
    st().apply("s1", { type: "text_delta", text: "hi" });
    st().apply("s2", { type: "started", run_id: "r2" });
    st().apply("s1", { type: "message", seq: 1, depth: 0, source_agent: null, message: asst("hi"), usage: null });

    expect(selectRun("s1")(st()).messagesDirty).toBe(true);
    expect(selectRun("s2")(st()).runId).toBe("r2");
    st().clearDirty("s1");
    expect(selectRun("s1")(st()).messagesDirty).toBe(false);

    st().reset("s1");
    expect(selectRun("s1")(st())).toEqual(emptyRun());
    // An unknown session reads as idle, and always as the same object: a selector that
    // built one per call would never settle in React.
    expect(selectRun("nope")(st())).toBe(selectRun(null)(st()));
  });
});
