import { describe, expect, it } from "vitest";

import {
  buildThread,
  groupDuration,
  liveGroupKey,
  namedTools,
  resolveCall,
  summarizeGroup,
  withLiveCalls,
  type GroupCall,
  type ResolvedCall,
  type Segment,
} from "./thread";
import type { ToolCallState } from "@/store/runs";
import type { Message, Part, StoredMessage } from "@/types";

let nextSeq = 0;

function stored(message: Message, depth = 0): StoredMessage {
  return { seq: nextSeq++, depth, source_agent: null, message, usage: null, created_at: 0 };
}

function fn(id: string, name: string, args: unknown = {}): Part {
  return { type: "function", id, function: { name, arguments: args } };
}

const user = (text: string) => stored({ role: "user", contents: [{ type: "text", text }] });
const said = (text: string, thinking?: string) =>
  stored({ role: "assistant", contents: [{ type: "text", text }], thinking });
const called = (calls: Part[], thinking?: string) =>
  stored({ role: "assistant", contents: [], tool_calls: calls, thinking });
const answered = (id: string, value: unknown) =>
  stored({ role: "tool", id, contents: [{ type: "value", value }] });

/** The shape of a thread, as one line per segment. */
function shape(segments: Segment[]): string[] {
  return segments.map((s) =>
    s.kind === "turn" ? `turn:${s.message.message.role}` : `tools:${s.calls.map((c) => c.name).join("+")}`,
  );
}

describe("buildThread", () => {
  it("runs the calls of adjacent messages into one group", () => {
    const { segments } = buildThread([
      user("go"),
      called([fn("a", "read")]),
      answered("a", { text: "ok" }),
      called([fn("b", "grep"), fn("c", "read")]),
      answered("b", { hits: 1 }),
      answered("c", { text: "ok" }),
      said("done"),
    ]);
    expect(shape(segments)).toEqual(["turn:user", "tools:read+grep+read", "turn:assistant"]);
  });

  it("closes a group on anything said, by either side", () => {
    const { segments } = buildThread([
      called([fn("a", "read")]),
      said("halfway there"),
      called([fn("b", "read")]),
      user("stop there"),
      called([fn("c", "read")]),
    ]);
    expect(shape(segments)).toEqual([
      "tools:read",
      "turn:assistant",
      "tools:read",
      "turn:user",
      "tools:read",
    ]);
  });

  it("splits a message that both says something and calls, text first", () => {
    const { segments } = buildThread([stored({ role: "assistant", contents: [{ type: "text", text: "looking" }], tool_calls: [fn("a", "read")] })]);
    expect(shape(segments)).toEqual(["turn:assistant", "tools:read"]);
  });

  it("puts the trace of a call-only message in the group, and leaves a turn's with the turn", () => {
    const { segments } = buildThread([
      called([fn("a", "read")], "first I read it"),
      called([fn("b", "grep")], "then I search"),
      said("found it", "and now I can say so"),
      called([fn("c", "read")], "one more"),
    ]);
    const [group, turn, second] = segments;
    expect(group.kind === "tools" && group.thinking).toEqual(["first I read it", "then I search"]);
    expect(turn.kind === "turn" && turn.message.message.thinking).toBe("and now I can say so");
    expect(second.kind === "tools" && second.thinking).toEqual(["one more"]);
  });

  it("keeps a trace out of the group when the same message says something", () => {
    // The turn draws it; a copy in the group beside it would be the same words twice.
    const { segments } = buildThread([
      stored({ role: "assistant", contents: [{ type: "text", text: "looking" }], tool_calls: [fn("a", "read")], thinking: "where is it" }),
    ]);
    expect(segments[1].kind === "tools" && segments[1].thinking).toEqual([]);
  });

  it("indexes tool answers by call id instead of drawing them", () => {
    const { segments, results } = buildThread([
      called([fn("a", "read")]),
      answered("a", { text: "ok" }),
      stored({ role: "tool", contents: [{ type: "value", value: 1 }] }), // no id: unmatchable
    ]);
    expect(shape(segments)).toEqual(["tools:read"]);
    expect([...results.keys()]).toEqual(["a"]);
  });

  it("hides sub-agent rows and the system message", () => {
    const { segments } = buildThread([
      stored({ role: "system", contents: [{ type: "text", text: "you are" }] }),
      stored({ role: "assistant", contents: [{ type: "text", text: "inner" }], tool_calls: [fn("z", "read")] }, 1),
      user("go"),
    ]);
    expect(shape(segments)).toEqual(["turn:user"]);
  });

  it("gives an empty message a turn of its own, so it draws as empty", () => {
    const { segments } = buildThread([stored({ role: "assistant", contents: [] })]);
    expect(shape(segments)).toEqual(["turn:assistant"]);
  });

  it("claims every id a stored message names", () => {
    const { claimed } = buildThread([called([fn("a", "read"), fn("b", "grep")]), called([fn("c", "read")])]);
    expect([...claimed]).toEqual(["a", "b", "c"]);
  });

  it("keys a group on its first call, so growing it does not rebuild it", () => {
    const first = buildThread([user("go"), called([fn("a", "read")])]).segments;
    const grown = buildThread([user("go"), called([fn("a", "read")]), called([fn("b", "grep")])]).segments;
    expect(first[1].key).toBe("tools:a");
    expect(grown[1].key).toBe("tools:a");
  });
});

describe("withLiveCalls", () => {
  const live: GroupCall[] = [{ id: "z", name: "shell", args: {} }];

  it("folds the calls onto the end of the trailing group", () => {
    const { segments } = buildThread([user("go"), called([fn("a", "read")])]);
    const out = withLiveCalls(segments, live);
    expect(shape(out)).toEqual(["turn:user", "tools:read+shell"]);
    // The stored segments are memoized upstream and must come back unchanged.
    expect(shape(segments)).toEqual(["turn:user", "tools:read"]);
  });

  it("opens a group when the thread ends in something said", () => {
    const { segments } = buildThread([user("go")]);
    expect(shape(withLiveCalls(segments, live))).toEqual(["turn:user", "tools:shell"]);
    expect(withLiveCalls(segments, live)[1].key).toBe("tools:z");
  });

  it("is nothing when there are no live calls", () => {
    const { segments } = buildThread([user("go")]);
    expect(withLiveCalls(segments, [])).toBe(segments);
  });
});

describe("liveGroupKey", () => {
  const ending = (messages: StoredMessage[]) => buildThread(messages).segments;

  it("is the group at the end of the thread while a run is going", () => {
    const segments = ending([user("go"), called([fn("a", "read")])]);
    expect(liveGroupKey(segments, true)).toBe("tools:a");
  });

  it("is nothing when no run is going", () => {
    const segments = ending([user("go"), called([fn("a", "read")])]);
    expect(liveGroupKey(segments, false)).toBeNull();
  });

  it("is nothing when the thread ends in something said, so an abandoned group stays quiet", () => {
    // The calls of a stopped turn are stored with no answers. Lighting them up again for
    // the next run would name a tool that stopped running an hour ago.
    const segments = ending([called([fn("a", "read")]), user("try again")]);
    expect(liveGroupKey(segments, true)).toBeNull();
  });

  it("is nothing for an empty thread", () => {
    expect(liveGroupKey([], true)).toBeNull();
  });
});

describe("resolveCall", () => {
  const call: GroupCall = { id: "a", name: "read", args: { path: "/a" } };
  const runningLive: ToolCallState = { id: "a", name: "read", arguments: {}, status: "running", startedAt: 1000 };

  it("takes the live entry where there is one, clock and all", () => {
    const out = resolveCall(call, runningLive, undefined, true);
    expect(out.status).toBe("running");
    expect(out.startedAt).toBe(1000);
  });

  it("prefers the live entry even once storage has an answer", () => {
    const done: ToolCallState = { ...runningLive, status: "done", result: { text: "live" }, finishedAt: 3000 };
    const out = resolveCall(call, done, answered("a", { text: "stored" }), false);
    expect(out.result).toEqual({ text: "live" });
    expect(out.finishedAt).toBe(3000);
  });

  it("reads a stored answer as done, and the engine's error shape as an error", () => {
    expect(resolveCall(call, undefined, answered("a", { text: "ok" }), false).status).toBe("done");
    const bad = resolveCall(call, undefined, answered("a", { error: "boom" }), false);
    expect(bad.status).toBe("error");
    expect(bad.result).toEqual({ error: "boom" });
  });

  it("reads the interruption stub as interrupted, and shows none of it", () => {
    const stub = stored({
      role: "tool",
      id: "a",
      contents: [{ type: "text", text: "[Interrupted: cancelled before this tool call completed]" }],
    });
    const out = resolveCall(call, undefined, stub, false);
    expect(out.status).toBe("interrupted");
    expect(out.result).toBeUndefined();
  });

  it("is running when the live group has no answer yet, and interrupted when the run is over", () => {
    // A reload mid-run lands in the first case, having missed the `tool_call_started`.
    expect(resolveCall(call, undefined, undefined, true).status).toBe("running");
    expect(resolveCall(call, undefined, undefined, false).status).toBe("interrupted");
  });
});

describe("the closed row", () => {
  const resolved = (name: string, status: ResolvedCall["status"]): ResolvedCall => ({
    id: name + status,
    name,
    args: {},
    status,
  });

  it("names each tool once and counts the rest past three", () => {
    expect(namedTools([{ id: "1", name: "read", args: {} }])).toBe("read");
    expect(namedTools(["read", "read", "grep"].map((n, i) => ({ id: `${i}`, name: n, args: {} })))).toBe("read, grep");
    expect(namedTools(["a", "b", "c", "d", "e"].map((n, i) => ({ id: `${i}`, name: n, args: {} })))).toBe("a, b, c +2");
  });

  it("says which call the agent is inside, and how many run beside it", () => {
    const s = summarizeGroup([
      resolved("read", "done"),
      resolved("shell", "running"),
      resolved("grep", "running"),
    ]);
    expect(s.active?.name).toBe("shell");
    expect(s.queued).toBe(1);
  });

  it("counts what a finished group has to answer for", () => {
    const s = summarizeGroup([
      resolved("read", "done"),
      resolved("shell", "error"),
      resolved("grep", "interrupted"),
      resolved("read", "interrupted"),
    ]);
    expect(s.active).toBeNull();
    expect({ errors: s.errors, interrupted: s.interrupted, queued: s.queued }).toEqual({
      errors: 1,
      interrupted: 2,
      queued: 0,
    });
  });

  it("times a group by the clock, not by the sum of its calls", () => {
    // Calls of one turn run together: two calls of two seconds each are two seconds.
    const a: ResolvedCall = { ...resolved("read", "done"), startedAt: 1_000, finishedAt: 3_000 };
    const b: ResolvedCall = { ...resolved("grep", "done"), startedAt: 1_200, finishedAt: 3_200 };
    expect(groupDuration([a, b])).toBe(2);
    // One call without the live run's clock withholds the number for the whole group.
    expect(groupDuration([a, resolved("shell", "done")])).toBeNull();
    expect(groupDuration([])).toBeNull();
  });
});
