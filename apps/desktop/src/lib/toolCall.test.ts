import { describe, expect, it } from "vitest";

import { describeCall, fieldsOf, groupHeadline, interruptedNote, isErrorResult, partialArg, pathRoots, previewArgs, previewLines, shellResult, shortenPaths, summarizeToolCall } from "./toolCall";

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

const INFO = {
  mountpoint: "/Users/me/Library/Application Support/com.brekkylab.ailoy/workspace",
  files_root: "/Users/me",
};
const ROOTS = pathRoots(INFO);

describe("a call as a line", () => {
  it("writes the workspace, the artifacts and the home directory by the names a reader knows", () => {
    expect(ROOTS.map((r) => r.label)).toEqual(["workspace", "artifacts", "~"]);
    expect(shortenPaths('ls -la "/Users/me/Library/Application Support/com.brekkylab.ailoy/workspace"', ROOTS)).toBe(
      "ls -la workspace",
    );
    expect(
      shortenPaths('cd "/Users/me/Library/Application Support/com.brekkylab.ailoy/workspace" && find . -type f', ROOTS),
    ).toBe("cd workspace && find . -type f");
    expect(shortenPaths("cat /Users/me/notes/todo.md", ROOTS)).toBe("cat ~/notes/todo.md");
    expect(
      shortenPaths('cp x "/Users/me/Library/Application Support/com.brekkylab.ailoy/artifacts/report.md"', ROOTS),
    ).toBe("cp x artifacts/report.md");
  });

  it("keeps a quote the path needs, and every quote that is the command's own", () => {
    expect(
      shortenPaths('cat "/Users/me/Library/Application Support/com.brekkylab.ailoy/workspace/My Notes/a.md"', ROOTS),
    ).toBe('cat "workspace/My Notes/a.md"');
    expect(shortenPaths('grep "workspace" file', ROOTS)).toBe('grep "workspace" file');
    // A path that only starts like a root is not under it.
    expect(shortenPaths("ls /Users/meadow", ROOTS)).toBe("ls /Users/meadow");
  });

  it("names a call by what it did", () => {
    expect(describeCall("shell", { cmd: "ls -la /Users/me" }, ROOTS)).toEqual({ verb: "Ran", target: "ls -la ~" });
    expect(describeCall("read", { path: "/Users/me/a.md" }, ROOTS)).toEqual({ verb: "Read", target: "~/a.md" });
    expect(describeCall("shell", { cmd: "set -e\ncd /tmp\nls" })).toEqual({ verb: "Ran", target: "set -e …" });
    expect(describeCall("mystery", { x: 1 }).verb).toBe("mystery");
    // Not finished, so not "Ran".
    expect(describeCall("shell", { cmd: "ls" }, [], true).verb).toBe("Running");
    expect(describeCall("read", { path: "a.md" }, [], true).verb).toBe("Reading");
    expect(describeCall("mystery", {}, [], true).verb).toBe("mystery");
  });

  it("sums a group up by what its calls did", () => {
    const calls = (...names: string[]) => names.map((name) => ({ name }));
    expect(groupHeadline(calls("shell", "shell", "shell"))).toBe("Ran 3 commands");
    expect(groupHeadline(calls("read", "shell", "read", "write", "edit"))).toBe("Read 2 files, ran 1 command, edited 2 files");
    expect(groupHeadline(calls("mystery", "other"))).toBe("Used 2 tools");
  });

  it("reads a shell answer the way a terminal shows it", () => {
    expect(shellResult({ stdout: "hi\n", stderr: "", exit_code: 0, timed_out: false, truncated: false })).toEqual({
      stdout: "hi\n",
      stderr: "",
      exitCode: 0,
      timedOut: false,
      truncated: false,
    });
    expect(shellResult({ error: "no console" })).toBeNull();
    expect(shellResult("text")).toBeNull();
  });

  it("previews the start of a long output and counts the rest", () => {
    const text = Array.from({ length: 25 }, (_, i) => `line ${i}`).join("\n") + "\n\n";
    const { shown, hidden } = previewLines(text);
    expect(shown.split("\n")).toHaveLength(20);
    expect(hidden).toBe(5);
    expect(previewLines("a\nb\n")).toEqual({ shown: "a\nb", hidden: 0 });
  });
});

describe("a call still being written", () => {
  it("reads as much of an argument as has arrived", () => {
    expect(partialArg('{"path":"/tmp/re', "path")).toBe("/tmp/re");
    expect(partialArg('{"path":"/tmp/report.md","content":"# Hi', "path")).toBe("/tmp/report.md");
    expect(partialArg('{"content":"# Hi","pa', "path")).toBeNull();
    expect(partialArg('{"n": 3', "n")).toBeNull();
    expect(partialArg('{ "cmd" : "echo \\"hi\\"\\n', "cmd")).toBe('echo "hi"\n');
    expect(partialArg('{"path":"caf\\u00e9', "path")).toBe("café");
    // Cut inside an escape: stop before it rather than show half of it.
    expect(partialArg('{"path":"caf\\u00', "path")).toBe("caf");
    expect(partialArg('{"path":"a\\', "path")).toBe("a");
  });

  it("previews only the argument that names the call", () => {
    expect(previewArgs("write", '{"path":"/Users/me/a.md","content":"long')).toEqual({ path: "/Users/me/a.md" });
    expect(previewArgs("write", '{"content":"long')).toBeUndefined();
    expect(previewArgs("shell", '{"cmd":"ls -l')).toEqual({ cmd: "ls -l" });
    expect(previewArgs("mystery", '{"x":"y"}')).toBeUndefined();
    expect(previewArgs("write", undefined)).toBeUndefined();
    // And a line with nothing to name it by yet is a verb on its own.
    expect(describeCall("write", undefined, [], true)).toEqual({ verb: "Writing", target: "" });
    // A path that has only reached part of the way to a root is not shown in the long form
    // it is about to lose; once past the root it is written short.
    const roots = pathRoots({ mountpoint: "/Users/me/Library/Application Support/x/workspace", files_root: "/Users/me" });
    expect(describeCall("write", { path: "/Users/me/Libr" }, roots, true).target).toBe("");
    expect(describeCall("write", { path: "/Users/me/Library/Application Support/x/workspace/a.md" }, roots, true).target).toBe(
      "workspace/a.md",
    );
    // A path that has left every root behind is shown as it is.
    expect(describeCall("write", { path: "/tmp/a" }, roots, true).target).toBe("/tmp/a");
  });
});
