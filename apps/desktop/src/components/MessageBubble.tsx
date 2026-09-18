// One stored turn: what the user said, or what the assistant said and did.
//
// A turn is not one row in the store. The assistant's message holds its `tool_calls`, and
// each call's answer is a *separate* stored message with `role: "tool"` whose `id` is the
// call id; `Thread` indexes those by id and hands the map down, so a call and its result
// render as the single card they read as.
//
// The assistant row is written when the model turn ends — before its tools run — so this
// bubble is on screen, listing calls, while those calls are still executing. `live` is how
// it tells a call that is *running* from one that was cut off: the run in flight is the
// only thing that knows.

import { Markdown } from "@/components/Markdown";
import { ToolCallCard } from "@/components/ToolCallCard";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { interruptedNote, isErrorResult } from "@/lib/toolCall";
import type { LiveRun, ToolStatus } from "@/store/runs";
import { S } from "@/strings";
import type { Message, StoredMessage } from "@/types";

function textOf(m: Message): string {
  return m.contents.map((p) => (p.type === "text" ? p.text : "")).join("");
}

/**
 * What the tool actually returned. Results are a single `value` part; the one exception
 * is the engine's interruption stub, which is a `text` part.
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
 * No stored answer at all means the run died before the engine could write one. An
 * answer that *is* the interruption stub means the same thing, written down — either way
 * the card says "Interrupted" and shows no output, because the stub is a marker for the model,
 * not a result the user wants to read.
 */
function statusOf(value: unknown, answered: boolean): ToolStatus {
  if (!answered || interruptedNote(value)) return "interrupted";
  return isErrorResult(value) ? "error" : "done";
}

export function UserBubble({ message }: { message: Message }) {
  return (
    <div className="ml-auto max-w-[80%] rounded-2xl bg-primary px-4 py-2 whitespace-pre-wrap text-primary-foreground">
      {textOf(message)}
    </div>
  );
}

export function AssistantBubble({
  message,
  toolResults,
  live,
  isLatest,
}: {
  message: Message;
  toolResults: Map<string, StoredMessage>;
  /** The current run for this session, idle or not. */
  live: LiveRun;
  /** Whether this is the newest assistant message — the only one a live run can be about. */
  isLatest: boolean;
}) {
  const text = textOf(message);
  return (
    <div className="max-w-[92%] space-y-1">
      {message.thinking && (
        <Collapsible>
          <CollapsibleTrigger className="text-xs text-muted-foreground hover:underline">
            {S.thinking}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="max-h-48 overflow-auto rounded bg-muted/40 p-2 text-xs whitespace-pre-wrap">
              {message.thinking}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      )}
      {text && <Markdown text={text} />}
      {(message.tool_calls ?? []).map((p) => {
        if (p.type !== "function") return null;
        // The live run wins where it has an entry: it is ahead of storage for the whole
        // stretch between the call starting and its result row being written, and it is
        // the only side that carries a start time to count from.
        const c = live.toolCalls[p.id];
        if (c) {
          return (
            <ToolCallCard
              key={p.id}
              name={p.function.name}
              args={p.function.arguments}
              status={c.status}
              result={c.result}
              startedAt={c.startedAt}
              finishedAt={c.finishedAt}
            />
          );
        }
        const stored = toolResults.get(p.id);
        const value = resultOf(stored);
        // No live entry and no result row. Still running if a run is going and this is the
        // turn it is working on — a reload mid-run lands here, having missed the
        // `tool_call_started`. Otherwise nothing will ever answer this call.
        const status =
          stored === undefined && live.status === "running" && isLatest
            ? "running"
            : statusOf(value, stored !== undefined);
        return (
          <ToolCallCard
            key={p.id}
            name={p.function.name}
            args={p.function.arguments}
            status={status}
            result={status === "done" || status === "error" ? value : undefined}
          />
        );
      })}
    </div>
  );
}
