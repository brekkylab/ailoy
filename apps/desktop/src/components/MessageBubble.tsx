// One stored turn: what the user said, or what the assistant said and did.
//
// A turn is not one row in the store. The assistant's message holds its `tool_calls`, and
// each call's answer is a *separate* stored message with `role: "tool"` whose `id` is the
// call id; `Thread` indexes those by id and hands the map down, so a call and its result
// render as the single card they read as.

import { Markdown } from "@/components/Markdown";
import { ToolCallCard } from "@/components/ToolCallCard";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { interruptedNote, isErrorResult } from "@/lib/toolCall";
import type { ToolStatus } from "@/store/runs";
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
 * the card says "중단됨" and shows no output, because the stub is a marker for the model,
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
}: {
  message: Message;
  toolResults: Map<string, StoredMessage>;
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
        const stored = toolResults.get(p.id);
        const value = resultOf(stored);
        const status = statusOf(value, stored !== undefined);
        return (
          <ToolCallCard
            key={p.id}
            name={p.function.name}
            args={p.function.arguments}
            status={status}
            result={status === "interrupted" ? undefined : value}
          />
        );
      })}
    </div>
  );
}
