// Reading a tool call for the eye rather than the model.
//
// A tool call is a name plus an arbitrary JSON payload, and its result is whatever the
// tool returned. The card has one line of room in its header and a panel underneath, so
// these three helpers decide what goes where: the one argument worth naming the call by,
// how a result object splits into fields, and whether a "result" is really the engine's
// note that the call never ran.

export type FieldKind = "inline" | "block";
export interface Field {
  key: string;
  value: string;
  kind: FieldKind;
}

/**
 * The argument that *is* the call, per tool. Everything else stays folded away; a tool
 * missing from here falls back to its whole argument object.
 */
const PRIMARY: Record<string, string> = {
  shell: "cmd",
  read: "path",
  write: "path",
  edit: "path",
  glob: "pattern",
  grep: "pattern",
  web_search: "query",
  web_fetch: "url",
};

export function summarizeToolCall(name: string, args: unknown): string {
  // A call still being written may not have its arguments yet.
  if (args === undefined) return "";
  if (args && typeof args === "object") {
    const key = PRIMARY[name];
    const v = key ? (args as Record<string, unknown>)[key] : undefined;
    if (typeof v === "string") return v;
  }
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}

/** A long or multi-line string needs its own `<pre>`; anything shorter reads on one line. */
const BLOCK_AT = 80;

export function fieldsOf(value: unknown): Field[] {
  if (value === null || value === undefined) return [];
  if (typeof value !== "object") return [{ key: "", value: String(value), kind: "block" }];
  if (Array.isArray(value)) return [{ key: "", value: JSON.stringify(value, null, 2), kind: "block" }];
  return Object.entries(value as Record<string, unknown>).map(([key, v]): Field => {
    if (typeof v === "string") {
      return { key, value: v, kind: v.includes("\n") || v.length > BLOCK_AT ? "block" : "inline" };
    }
    if (typeof v === "number" || typeof v === "boolean") return { key, value: String(v), kind: "inline" };
    return { key, value: JSON.stringify(v, null, 2), kind: "block" };
  });
}

/**
 * A tool result that reports failure. The engine wraps a tool's own error as
 * `{ error: … }`, which is the one shape the card paints red.
 */
export function isErrorResult(value: unknown): boolean {
  return typeof value === "object" && value !== null && "error" in (value as Record<string, unknown>);
}

/**
 * Whether a stored tool message is the engine's stub for a call that never answered.
 *
 * When a run is cancelled or fails mid-batch, `close_dangling_tool_calls`
 * (`src/agent/rt.rs`) appends one Tool message per unanswered call whose single part is
 * *text* — `Part::text(note)`, not a value part — carrying
 * `"[Interrupted: cancelled before this tool call completed]"` or
 * `"[Interrupted: tool execution failed before this tool call completed]"`. It is a
 * marker for the model, not a result: the card shows it as interrupted rather than as
 * output, and never as an error.
 */
const INTERRUPTED_PREFIX = "[Interrupted:";

export function interruptedNote(value: unknown): boolean {
  return typeof value === "string" && value.trimStart().startsWith(INTERRUPTED_PREFIX);
}

// ── How a call reads as a line ─────────────────────────────────────────────

/** A directory the agent names by its absolute path, and the word a reader knows it by. */
export interface PathRoot {
  path: string;
  label: string;
}

/**
 * The roots worth shortening, from what the engine reports about the workspace.
 *
 * The agent names everything by absolute path — the session stands in its scratch tree, so
 * a relative one would read the wrong directory — and the paths are long: every command it
 * ran through the workspace began `cd "/Users/…/Application Support/com.brekkylab.ailoy/
 * workspace"`. Longest first, because the workspace sits under the home directory and has
 * to be matched before it.
 */
export function pathRoots(info: { mountpoint: string; files_root: string } | undefined): PathRoot[] {
  if (!info) return [];
  const dataDir = info.mountpoint.replace(/\/[^/]+\/?$/, "");
  const roots: PathRoot[] = [
    { path: info.mountpoint, label: "workspace" },
    { path: `${dataDir}/artifacts`, label: "artifacts" },
    { path: info.files_root, label: "~" },
  ];
  return roots.filter((r) => r.path.length > 1).sort((a, b) => b.path.length - a.path.length);
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * `text` with each root written as its label — `"/Users/…/workspace/notion"` becomes
 * `workspace/notion`. The quotes a path needed for its spaces are dropped with it when what
 * is left has none, and only then: a quoted argument that is not one of these paths is the
 * command's own, and changing it would misreport what ran.
 */
export function shortenPaths(text: string, roots: PathRoot[]): string {
  // Each replacement is marked until the quotes have been looked at, so a quoted word that
  // merely *reads* like a label — `grep "workspace"` — is never taken for one.
  const MARK = "\u0000";
  let out = text;
  for (const { path, label } of roots) {
    const bare = path.replace(/\/+$/, "");
    out = out.replace(new RegExp(`${escapeRe(bare)}(?=/|"|'|\\s|$)`, "g"), MARK + label);
  }
  out = out.replace(new RegExp(`(["'])(${MARK}[^"'\\s]*)\\1`, "g"), "$2");
  return out.split(MARK).join("");
}

/** What a call did, as a verb, and what it did it to. */
export interface CallLine {
  verb: string;
  target: string;
}

/** Each tool's verb, as it reads once the call is over and while it is still going. */
const VERBS: Record<string, { done: string; running: string }> = {
  shell: { done: "Ran", running: "Running" },
  read: { done: "Read", running: "Reading" },
  write: { done: "Wrote", running: "Writing" },
  edit: { done: "Edited", running: "Editing" },
  glob: { done: "Found", running: "Finding" },
  grep: { done: "Searched", running: "Searching" },
  web_search: { done: "Searched the web", running: "Searching the web" },
  web_fetch: { done: "Fetched", running: "Fetching" },
};

/**
 * One call as the line a reader scans: `Ran` `ls -la workspace`, `Read` `~/notes.md` — or,
 * while it is still going, `Running`, `Reading`: a call that has not finished must not say
 * it has. A command of several lines shows its first, with an ellipsis — the whole of it is
 * in the panel. A tool this does not know keeps its own name as the verb.
 */
export function describeCall(
  name: string,
  args: unknown,
  roots: PathRoot[] = [],
  running = false,
): CallLine {
  const raw = summarizeToolCall(name, args);
  // A path still arriving that is so far only the start of a root — `/Users/me/Libr` on its
  // way to the workspace — would flash by as the long absolute path it is about to stop
  // being. It is shown as nothing yet until it is long enough to be written short.
  if (running && raw && roots.some((r) => r.path.length > raw.length && r.path.startsWith(raw))) {
    return { verb: VERBS[name]?.running ?? name, target: "" };
  }
  const full = shortenPaths(raw, roots);
  const lines = full.split("\n").filter((l) => l.trim());
  const target = lines.length > 1 ? `${lines[0].trim()} …` : (lines[0] ?? "").trim();
  const verb = VERBS[name];
  return { verb: verb ? (running ? verb.running : verb.done) : name, target };
}

/** How each kind of call is counted in a group's headline: the phrase, singular and plural. */
const COUNTED: Record<string, [string, string, string]> = {
  shell: ["Ran", "command", "commands"],
  read: ["Read", "file", "files"],
  write: ["Edited", "file", "files"],
  edit: ["Edited", "file", "files"],
  glob: ["Searched", "time", "times"],
  grep: ["Searched", "time", "times"],
  web_search: ["Searched the web", "time", "times"],
  web_fetch: ["Fetched", "page", "pages"],
};

/**
 * A run of calls in a phrase: "Ran 13 commands", "Read 2 files, ran 3 commands". Counted by
 * what they did rather than by tool, so `write` and `edit` are one "Edited", in the order
 * each first happened. A tool this does not know is counted as "used N tools".
 */
export function groupHeadline(calls: { name: string }[]): string {
  const counts = new Map<string, { phrase: [string, string, string]; n: number }>();
  for (const c of calls) {
    const phrase = COUNTED[c.name] ?? (["Used", "tool", "tools"] as [string, string, string]);
    const key = `${phrase[0]} ${phrase[2]}`;
    const entry = counts.get(key) ?? { phrase, n: 0 };
    entry.n += 1;
    counts.set(key, entry);
  }
  return [...counts.values()]
    .map(({ phrase: [verb, one, many], n }, i) => `${i === 0 ? verb : verb.toLowerCase()} ${n} ${n === 1 ? one : many}`)
    .join(", ");
}

/** A `shell` result, as the terminal it came from would show it. */
export interface ShellResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  truncated: boolean;
}

/** `null` for anything that is not the shape the `shell` tool answers in. */
export function shellResult(value: unknown): ShellResult | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const v = value as Record<string, unknown>;
  if (typeof v.stdout !== "string" && typeof v.stderr !== "string") return null;
  return {
    stdout: typeof v.stdout === "string" ? v.stdout : "",
    stderr: typeof v.stderr === "string" ? v.stderr : "",
    exitCode: typeof v.exit_code === "number" ? v.exit_code : null,
    timedOut: v.timed_out === true,
    truncated: v.truncated === true,
  };
}

/** How many lines of output a panel shows before it offers the rest. */
export const PREVIEW_LINES = 20;

/** The first `max` lines of `text`, and how many were left out. Trailing blank lines do not count. */
export function previewLines(text: string, max = PREVIEW_LINES): { shown: string; hidden: number } {
  const lines = text.replace(/\n+$/, "").split("\n");
  if (lines.length <= max) return { shown: lines.join("\n"), hidden: 0 };
  return { shown: lines.slice(0, max).join("\n"), hidden: lines.length - max };
}

// ── A call the model is still writing ──────────────────────────────────────

/**
 * The value of `key` in a JSON object that has not finished arriving — as much of it as is
 * there. `{"path":"/tmp/re` answers `/tmp/re` for `path`; a key not reached yet, or one
 * whose value is not a string, answers `null`. Escapes are decoded as far as they are
 * complete, so a value cut in the middle of `\u00e9` stops before it rather than showing
 * the half.
 *
 * Only the top level is looked at, and only well enough for a preview: the finished call
 * replaces all of this with the real arguments.
 */
export function partialArg(json: string, key: string): string | null {
  const head = new RegExp(`"${escapeRe(key)}"\\s*:\\s*"`).exec(json);
  if (!head) return null;
  let out = "";
  for (let i = head.index + head[0].length; i < json.length; i++) {
    const c = json[i];
    if (c === '"') return out;
    if (c !== "\\") {
      out += c;
      continue;
    }
    const next = json[i + 1];
    if (next === undefined) break;
    if (next === "u") {
      const hex = json.slice(i + 2, i + 6);
      if (hex.length < 4) break;
      out += String.fromCharCode(parseInt(hex, 16));
      i += 5;
      continue;
    }
    out += ({ n: "\n", t: "\t", r: "\r", b: "\b", f: "\f" } as Record<string, string>)[next] ?? next;
    i += 1;
  }
  return out;
}

/**
 * What can be shown of a call's arguments before they are complete: the one argument that
 * names the call (`PRIMARY`), once any of it has arrived. `undefined` until then, which
 * `describeCall` draws as a call with no target yet.
 */
export function previewArgs(name: string, argsText: string | undefined): unknown {
  const key = PRIMARY[name];
  if (!key || !argsText) return undefined;
  const value = partialArg(argsText, key);
  return value === null ? undefined : { [key]: value };
}
