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
