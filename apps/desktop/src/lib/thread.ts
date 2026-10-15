// The conversation, split into what was said and the runs of calls between.
//
// A thread is mostly tool calls. A turn that reads a file, greps for a symbol and then
// reads three more is five rows of chrome around one sentence of prose, and what a reader
// scrolling past them wants is one line saying the agent was working — not five lines to
// scroll over on the way to the sentence that came of them. So adjacent calls collapse
// into one group, and the rule for "adjacent" is the boundary a reader actually perceives:
// **saying something closes the group.** A message carrying both text and calls splits in
// two, text first, and the calls open a group of their own.
//
// Thinking is not saying. A model that reasons before each of twenty calls stores twenty
// messages carrying a trace and a call, and closing the group on those would draw twenty
// groups of one — more rows than there were before any of this. A trace from a message
// that only called tools goes *into* the group instead, which is also where it belongs: it
// is addressed to the agent, like everything else behind that line. A trace from a message
// that said something stays with the turn, which draws it.
//
// All of this reads the stored list and nothing else, so it is tested without a webview.
// What `Thread` adds on top is the live run, which is ahead of storage for the whole
// stretch between a call starting and its result row being written — see `resolveCall`.

import { interruptedNote, isErrorResult, previewArgs } from "@/lib/toolCall";
import type { ToolCallState, ToolStatus } from "@/store/runs";
import type { Message, StoredMessage } from "@/types";

/** One call as a group draws it: the name on the row, and the arguments behind it. */
export interface GroupCall {
  id: string;
  name: string;
  args: unknown;
}

/**
 * One stretch of the conversation, as a reader meets it.
 *
 * `key` is what React keeps a row by. A turn is keyed on its `seq`, which never moves
 * because the list is append-only; a group is keyed on its *first* call, which does not
 * change as the group grows — so a group the reader opened stays open, and stays the same
 * component, while the run keeps adding calls to it.
 */
export type Segment =
  | { kind: "turn"; key: string; message: StoredMessage }
  | { kind: "tools"; key: string; calls: GroupCall[]; thinking: string[] };

/** The stored conversation, ready to draw. */
export interface ThreadView {
  segments: Segment[];
  /** What the tools answered, by the id of the call each answer belongs to. */
  results: Map<string, StoredMessage>;
  /**
   * Every call id a stored assistant message names. The seam with the live run: a call
   * the engine has announced but storage has not claimed yet is drawn by the live block,
   * so there is exactly one card per call at every moment of a run.
   */
  claimed: Set<string>;
}

/** What a turn reads as, with the parts this thread cannot draw left out. */
export function textOf(m: Message): string {
  return m.contents.map((p) => (p.type === "text" ? p.text : "")).join("");
}

/** The calls a message asked for, in the order it asked for them. */
export function callsOf(m: Message): GroupCall[] {
  const out: GroupCall[] = [];
  for (const p of m.tool_calls ?? []) {
    if (p.type === "function") out.push({ id: p.id, name: p.function.name, args: p.function.arguments });
  }
  return out;
}

export function buildThread(messages: StoredMessage[]): ThreadView {
  const results = new Map<string, StoredMessage>();
  const claimed = new Set<string>();
  const segments: Segment[] = [];
  for (const m of messages) {
    if (m.depth !== 0) continue; // sub-agent internals stay hidden in v1
    const msg = m.message;
    // A tool row is an answer to a call, never a turn of its own. One without an `id`
    // cannot be matched to its call, but it is still not a bubble.
    if (msg.role === "tool") {
      if (msg.id) results.set(msg.id, m);
      continue;
    }
    if (msg.role === "system") continue;

    const calls = callsOf(msg);
    for (const c of calls) claimed.add(c.id);
    const said = textOf(msg).trim();
    const thought = (msg.thinking ?? "").trim();
    // A message with neither text nor calls still gets a turn, so an empty one draws as
    // empty rather than vanishing.
    if (said || !calls.length) segments.push({ kind: "turn", key: `turn:${m.seq}`, message: m });
    if (!calls.length) continue;

    // Not `said` again: the turn just pushed is itself what closes the group, so asking
    // what the last segment is answers both cases at once. It also settles where the trace
    // goes — reaching a group here means no turn was pushed, so nothing else draws it.
    const last = segments[segments.length - 1];
    if (last?.kind === "tools") {
      last.calls = [...last.calls, ...calls];
      if (thought) last.thinking = [...last.thinking, thought];
    } else {
      segments.push({
        kind: "tools",
        key: `tools:${calls[0].id || m.seq}`,
        calls,
        // `!said` because the turn that closed the last group is this same message, and
        // its bubble draws its trace. Reaching the branch above instead means no turn was
        // pushed, so there `!said` is already true.
        thinking: thought && !said ? [thought] : [],
      });
    }
  }
  return { segments, results, claimed };
}

/**
 * The calls the engine has announced but no stored message names yet, put where they
 * belong: on the end of the trailing group, or in one of their own.
 *
 * The caller decides *whether* to do this, because it is only right while the live block
 * above them is empty. Streamed text belongs after everything stored and before the calls
 * the model asked for at the end of it, and folding those calls into a stored group would
 * draw them above the text they came after.
 */
export function withLiveCalls(segments: Segment[], calls: GroupCall[]): Segment[] {
  if (!calls.length) return segments;
  const last = segments[segments.length - 1];
  if (last?.kind === "tools") {
    return [...segments.slice(0, -1), { ...last, calls: [...last.calls, ...calls] }];
  }
  return [...segments, { kind: "tools", key: `tools:${calls[0].id}`, calls, thinking: [] }];
}

/**
 * Which group the live run is inside, by key, and `null` for none.
 *
 * Not "is something running" — that is one flag on the session — but "is *this* group the
 * one it is running in", and the two come apart the moment a reader stops a turn. A
 * stopped run leaves calls nothing ever answered; asked the loose question, every later
 * run would light that old group up again and name a tool that stopped running an hour
 * ago. The group in flight is the one at the end of the thread: a run reaches a later
 * group only once the calls of the one before it have been answered, and anything the
 * model says closes the group it says it after.
 */
export function liveGroupKey(segments: Segment[], running: boolean): string | null {
  if (!running) return null;
  const last = segments[segments.length - 1];
  return last?.kind === "tools" ? last.key : null;
}

/** A call and how it ended — what a card needs and where each part came from. */
export interface ResolvedCall extends GroupCall {
  status: ToolStatus;
  /** The model is still writing the arguments; `args` is a preview of them. See `ToolCallState`. */
  preparing?: boolean;
  result?: unknown;
  /**
   * When the call began and ended. The live run's own clock while it has one; read back from
   * storage, when its result row says the call began and when that row was written — absent
   * for a row from before the engine kept it.
   */
  startedAt?: number;
  finishedAt?: number;
}

/**
 * What the tool actually returned. Results are a single `value` part; the one exception is
 * the engine's interruption stub, which is a `text` part.
 */
function resultOf(stored: StoredMessage | undefined): unknown {
  const first = stored?.message.contents[0];
  if (!first) return undefined;
  if (first.type === "value") return first.value;
  if (first.type === "text") return first.text;
  return undefined;
}

/**
 * How a call ended, read back from storage.
 *
 * No stored answer at all means the run died before the engine could write one. An answer
 * that *is* the interruption stub means the same thing, written down — either way the card
 * says "Interrupted" and shows no output, because the stub is a marker for the model, not
 * a result the user wants to read.
 */
function storedStatus(value: unknown, answered: boolean): ToolStatus {
  if (!answered || interruptedNote(value)) return "interrupted";
  return isErrorResult(value) ? "error" : "done";
}

/**
 * One call, as the three sources of truth leave it.
 *
 * The live run wins where it has an entry: it is ahead of storage for the whole stretch
 * between the call starting and its result row being written, and it is the only side that
 * carries a start time to count from. With no live entry and no result row the call is
 * still running if `live` — a reload mid-run lands here, having missed the
 * `tool_call_started` — and otherwise nothing will ever answer it.
 */
export function resolveCall(
  call: GroupCall,
  liveCall: ToolCallState | undefined,
  stored: StoredMessage | undefined,
  live: boolean,
): ResolvedCall {
  if (liveCall) {
    return {
      ...call,
      // What there is of the arguments while they are written — the one that names the call.
      args: liveCall.preparing ? previewArgs(call.name, liveCall.argsText) : call.args,
      preparing: liveCall.preparing,
      status: liveCall.status,
      result: liveCall.result,
      startedAt: liveCall.startedAt,
      finishedAt: liveCall.finishedAt,
    };
  }
  const value = resultOf(stored);
  const status = stored === undefined && live ? "running" : storedStatus(value, stored !== undefined);
  const clock =
    stored?.started_at != null ? { startedAt: stored.started_at, finishedAt: stored.created_at } : {};
  return { ...call, ...clock, status, result: status === "done" || status === "error" ? value : undefined };
}

/** How many names a closed row lists before it counts the rest instead. */
const NAMED = 3;

/**
 * The tools a group used, named once each and cut to three. A closed row is one line: past
 * three names it counts the rest rather than listing them.
 */
export function namedTools(calls: GroupCall[]): string {
  const seen = [...new Set(calls.map((c) => c.name))];
  const rest = seen.length - NAMED;
  return seen.slice(0, NAMED).join(", ") + (rest > 0 ? ` +${rest}` : "");
}

/** What a closed group says about itself. */
export interface GroupSummary {
  /** The call the agent is inside right now, if any. */
  active: ResolvedCall | null;
  /** How many more are running behind it. */
  queued: number;
  errors: number;
  interrupted: number;
}

export function summarizeGroup(calls: ResolvedCall[]): GroupSummary {
  const running = calls.filter((c) => c.status === "running");
  return {
    active: running[0] ?? null,
    queued: Math.max(0, running.length - 1),
    errors: calls.filter((c) => c.status === "error").length,
    interrupted: calls.filter((c) => c.status === "interrupted").length,
  };
}

/**
 * How long a finished group took, in seconds, or `null` when it cannot be known.
 *
 * Wall clock and not the sum of the calls: the engine runs a turn's calls together, so
 * three calls of two seconds each are two seconds of waiting, not six. The live run carries
 * the timestamps, and a stored call carries them on its result row (`started_at`, and the
 * row's own `created_at`); a row from before the engine kept them has none, and one call
 * without them is enough to withhold the number for the whole group rather than guess.
 */
export function groupDuration(calls: ResolvedCall[]): number | null {
  if (!calls.length) return null;
  let first = Infinity;
  let last = -Infinity;
  for (const c of calls) {
    if (c.startedAt == null || c.finishedAt == null) return null;
    first = Math.min(first, c.startedAt);
    last = Math.max(last, c.finishedAt);
  }
  return Math.max(0, Math.round((last - first) / 1000));
}

/** What a finished answer's actions work on: its prose, and when it was done. */
export interface TurnEnd {
  text: string;
  at: number;
}

/**
 * Where each answer's actions go, keyed on the segment that ends it.
 *
 * An answer is everything the assistant said between one user message and the next — often
 * a line, a run of tool calls, then the reply — and it gets its copy button and its time
 * once, under the last thing it said, the way the chat apps draw one reply as one thing.
 * Copying takes the whole of its prose, not only the last line of it. A reply that is still
 * being written (`open`, for the last one) gets none yet: it is not finished, and the time it
 * would show is not when it was.
 */
export function turnEnds(segments: Segment[], open = false): Map<string, TurnEnd> {
  const out = new Map<string, TurnEnd>();
  let texts: string[] = [];
  let last: { key: string; at: number } | null = null;
  const close = () => {
    if (last) out.set(last.key, { text: texts.join("\n\n"), at: last.at });
    texts = [];
    last = null;
  };
  for (const seg of segments) {
    if (seg.kind !== "turn") continue;
    const { message, created_at } = seg.message;
    if (message.role === "user") {
      close();
    } else if (message.role === "assistant") {
      const text = textOf(message);
      if (text.trim()) {
        texts.push(text);
        last = { key: seg.key, at: created_at };
      }
    }
  }
  if (!open) close();
  return out;
}
