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
  it("shows a call from when it is named, and swaps the preview for the real arguments", () => {
    const writing = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_preparing", id: "c1", name: "write" },
      { type: "tool_call_args_delta", id: "c1", chunk: '{"path":"/tmp/' },
      { type: "tool_call_args_delta", id: "c1", chunk: 'a.md","content":"' },
    ]);
    expect(writing.toolOrder).toEqual(["c1"]);
    expect(writing.toolCalls.c1).toMatchObject({ name: "write", status: "running", preparing: true, arguments: undefined });
    expect(writing.toolCalls.c1.argsText).toBe('{"path":"/tmp/a.md","content":"');

    const started = runFrom(writing, [
      { type: "tool_call_started", id: "c1", name: "write", arguments: { path: "/tmp/a.md", content: "hi" } },
      // A late chunk for a call that has started is dropped: the arguments are final.
      { type: "tool_call_args_delta", id: "c1", chunk: "noise" },
    ]);
    expect(started.toolOrder).toEqual(["c1"]);
    expect(started.toolCalls.c1.preparing).toBeUndefined();
    expect(started.toolCalls.c1.argsText).toBeUndefined();
    expect(started.toolCalls.c1.arguments).toEqual({ path: "/tmp/a.md", content: "hi" });
    // One clock from when it was named to when it ends, so the time it reports is the time
    // the call took — writing the file out included.
    expect(started.toolCalls.c1.startedAt).toBe(writing.toolCalls.c1.startedAt);
  });

  it("marks a call cancelled while it was being written as interrupted", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_preparing", id: "c1", name: "write" },
      { type: "cancelled" },
    ]);
    expect(s.toolCalls.c1.status).toBe("interrupted");
  });

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

  it("clears live text when the assistant message is persisted and bumps the message version", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "text_delta", text: "Hello" },
      { type: "message", seq: 2, depth: 0, source_agent: null, message: asst("Hello"), usage: null },
    ]);
    expect(s.text).toBe("");
    expect(s.messagesVersion).toBe(1);
    expect(s.messagesAcked).toBe(0);
  });

  it("counts every persisted message, tool answers included", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "ls" } },
      { type: "message", seq: 2, depth: 0, source_agent: null, message: asst("", [{ type: "function", id: "c1", function: { name: "shell", arguments: { cmd: "ls" } } }]), usage: null },
      { type: "message", seq: 3, depth: 0, source_agent: null, message: tool("c1", { stdout: "a\n" }), usage: null },
      { type: "message", seq: 4, depth: 1, source_agent: "sub", message: asst("nested"), usage: null },
    ]);
    expect(s.messagesVersion).toBe(3);
  });

  it("carries the message counters across a new run", () => {
    const s = runFrom(run([{ type: "started", run_id: "r1" }, { type: "message", seq: 1, depth: 0, source_agent: null, message: asst("hi"), usage: null }]), [
      { type: "done" },
      { type: "started", run_id: "r2" },
    ]);
    // They describe the stored list, not the run; rewinding them under a thread that has
    // already acked 1 would read as "behind" with nothing to fetch.
    expect(s.messagesVersion).toBe(1);
    expect(s.text).toBe("");
    expect(s.runId).toBe("r2");
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

  it("paints a tool result carrying an error key as a failure", () => {
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "nope" } },
      { type: "message", seq: 3, depth: 0, source_agent: null, message: tool("c1", { error: "no such command" }), usage: null },
      { type: "tool_call_started", id: "c2", name: "shell", arguments: { cmd: "ls" } },
      // A non-zero exit is the tool answering, not the tool failing.
      { type: "message", seq: 4, depth: 0, source_agent: null, message: tool("c2", { stdout: "", exit_code: 1 }), usage: null },
    ]);
    expect(s.toolCalls.c1.status).toBe("error");
    expect(s.toolCalls.c2.status).toBe("done");
  });

  it("keeps one entry in toolOrder when a call is announced twice", () => {
    // A re-attach can replay an announcement the store already has.
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "ls" } },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: { cmd: "ls -la" } },
    ]);
    expect(s.toolOrder).toEqual(["c1"]);
    expect(s.toolCalls.c1.arguments).toEqual({ cmd: "ls -la" });
  });

  it("does not interrupt a still-running tool when the run finishes normally", () => {
    // `done` is the engine's ordinary end; only cancel and error cut a call short.
    const s = run([
      { type: "started", run_id: "r1" },
      { type: "tool_call_started", id: "c1", name: "shell", arguments: {} },
      { type: "done" },
    ]);
    expect(s.status).toBe("done");
    expect(s.toolCalls.c1.status).toBe("running");
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
  it("keeps one run per session and tracks the refetch version", () => {
    const st = () => useRunStore.getState();
    st().apply("s1", { type: "started", run_id: "r1" });
    st().apply("s1", { type: "text_delta", text: "hi" });
    st().apply("s2", { type: "started", run_id: "r2" });
    st().apply("s1", { type: "message", seq: 1, depth: 0, source_agent: null, message: asst("hi"), usage: null });

    expect(selectRun("s1")(st()).messagesVersion).toBe(1);
    expect(selectRun("s2")(st()).runId).toBe("r2");
    st().ackMessages("s1", 1);
    expect(selectRun("s1")(st()).messagesAcked).toBe(1);

    // A message that lands while the refetch for version 1 is in flight: acking 1 when it
    // resolves leaves the pair mismatched, which is what schedules the next round.
    st().apply("s1", { type: "message", seq: 2, depth: 0, source_agent: null, message: asst("more"), usage: null });
    st().ackMessages("s1", 1);
    expect(selectRun("s1")(st())).toMatchObject({ messagesVersion: 2, messagesAcked: 1 });

    // An out-of-order ack never walks the mark backwards.
    st().ackMessages("s1", 2);
    st().ackMessages("s1", 1);
    expect(selectRun("s1")(st()).messagesAcked).toBe(2);

    st().reset("s1");
    expect(selectRun("s1")(st())).toEqual(emptyRun());
    // An unknown session reads as idle, and always as the same object: a selector that
    // built one per call would never settle in React.
    expect(selectRun("nope")(st())).toBe(selectRun(null)(st()));
  });
});
