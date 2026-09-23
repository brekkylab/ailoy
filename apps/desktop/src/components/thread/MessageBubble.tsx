// One stored turn: what the user said, or what the assistant said.
//
// What the assistant *did* is not here. A turn's calls are drawn by the group they fall
// into, which is a stretch of the conversation rather than a message — several messages'
// calls run together into one group, and the answers arrive as separate stored messages
// again. `lib/thread` cuts the conversation into those stretches and `Thread` hands each
// one to `ToolGroup`; a bubble draws the prose, and the trace that came with it.

import { Markdown } from "@/components/Markdown";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { textOf } from "@/lib/thread";
import { S } from "@/strings";
import type { Message } from "@/types";

export function UserBubble({ message }: { message: Message }) {
  return (
    // A tint of the text colour rather than the brand colour: lighter than the page in the
    // dark theme and darker in the light one, and quiet in both. The accent is kept for the
    // send button, so the most colourful thing on screen is not what the user already said.
    <div className="ml-auto max-w-[80%] rounded-2xl bg-foreground/[0.07] px-4 py-2.5 whitespace-pre-wrap">
      {textOf(message)}
    </div>
  );
}

export function AssistantBubble({ message }: { message: Message }) {
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
    </div>
  );
}
