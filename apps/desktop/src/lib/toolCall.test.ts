import { describe, expect, it } from "vitest";

import { fieldsOf, interruptedNote, isErrorResult, summarizeToolCall } from "./toolCall";

describe("toolCall helpers", () => {
  it("summarizes known tools by their primary argument", () => {
    expect(summarizeToolCall("shell", { cmd: "ls -la" })).toBe("ls -la");
    expect(summarizeToolCall("read", { path: "/a.txt" })).toBe("/a.txt");
    expect(summarizeToolCall("web_search", { query: "rust" })).toBe("rust");
    expect(summarizeToolCall("unknown", { a: 1 })).toBe('{"a":1}');
  });

  it("falls back to the whole payload when the primary argument is missing or not text", () => {
    expect(summarizeToolCall("shell", { timeout: 5 })).toBe('{"timeout":5}');
    expect(summarizeToolCall("read", { path: 7 })).toBe('{"path":7}');
    // A payload JSON cannot express still has to produce a header line.
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(summarizeToolCall("shell", cyclic)).toBe("[object Object]");
  });

  it("splits an object result into displayable fields", () => {
    const f = fieldsOf({ stdout: "line1\nline2", exit_code: 0, truncated: false });
    expect(f.map((x) => x.key)).toEqual(["stdout", "exit_code", "truncated"]);
    expect(f[0].kind).toBe("block");
    expect(f[1].kind).toBe("inline");
    expect(fieldsOf("plain")).toEqual([{ key: "", value: "plain", kind: "block" }]);
  });

  it("blocks long strings, keeps short ones inline, and pretty-prints nested values", () => {
    const f = fieldsOf({ short: "ok", long: "x".repeat(81), nested: { a: 1 } });
    expect(f[0].kind).toBe("inline");
    expect(f[1].kind).toBe("block");
    expect(f[2]).toEqual({ key: "nested", value: '{\n  "a": 1\n}', kind: "block" });
    expect(fieldsOf(null)).toEqual([]);
    expect(fieldsOf(undefined)).toEqual([]);
    expect(fieldsOf([1, 2])).toEqual([{ key: "", value: "[\n  1,\n  2\n]", kind: "block" }]);
  });

  it("recognizes the engine's error shape", () => {
    expect(isErrorResult({ error: "boom" })).toBe(true);
    expect(isErrorResult({ stdout: "", exit_code: 1 })).toBe(false);
    expect(isErrorResult("error")).toBe(false);
    expect(isErrorResult(null)).toBe(false);
  });

  it("recognizes the stub left for a tool call that never answered", () => {
    // Verbatim from `close_dangling_tool_calls` in `src/agent/rt.rs`.
    expect(interruptedNote("[Interrupted: cancelled before this tool call completed]")).toBe(true);
    expect(interruptedNote("[Interrupted: tool execution failed before this tool call completed]")).toBe(true);
    // Real output that merely talks about interruption is not the marker.
    expect(interruptedNote("the job was [Interrupted: ...]")).toBe(false);
    expect(interruptedNote({ stdout: "[Interrupted:" })).toBe(false);
    expect(interruptedNote(undefined)).toBe(false);
  });
});
