// One stored turn: what the user said, or what the assistant said.
//
// What the assistant *did* is not here. A turn's calls are drawn by the group they fall
// into, which is a stretch of the conversation rather than a message — several messages'
// calls run together into one group, and the answers arrive as separate stored messages
// again. `lib/thread` cuts the conversation into those stretches and `Thread` hands each
// one to `ToolGroup`; a bubble draws the prose, and the trace that came with it.

import { cn } from "cn";
import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { copyText } from "@/lib/clipboard";
import { textOf, type TurnEnd } from "@/lib/thread";
import { formatElapsed } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import { S } from "@/strings";
import type { Message } from "@/types";

/**
 * What can be done with a message once it is written: copy it, and see how long ago it was
 * (the exact time on hover). Under a
 * user's message, and once under a finished answer — not under every line the assistant
 * said on the way to it. Only while hovered or focused, the way the chat apps keep a thread
 * free of controls nobody is reaching for.
 */
function MessageActions({ text, at, end }: { text: string; at?: number; end?: boolean }) {
  const [copied, setCopied] = useState(false);
  const now = useNow();
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  return (
    <div
      className={cn(
        "flex items-center gap-1 text-muted-foreground opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100",
        end && "justify-end",
      )}
    >
      <button
        className="rounded-md p-1 hover:bg-accent hover:text-foreground"
        aria-label={copied ? S.copied : S.copy}
        title={copied ? S.copied : S.copy}
        onClick={() => void copyText(text).then(setCopied)}
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </button>
      {at != null && (
        <time className="px-1 text-[11px] tabular-nums" dateTime={new Date(at).toISOString()} title={new Date(at).toLocaleString()}>
          {formatElapsed(at, now)}
        </time>
      )}
    </div>
  );
}

export function UserBubble({ message, at }: { message: Message; at?: number }) {
  const text = textOf(message);
  return (
    <div className="group/msg ml-auto flex max-w-[80%] flex-col items-end gap-1">
      {/* A tint of the text colour rather than the brand colour: lighter than the page in the
          dark theme and darker in the light one, and quiet in both. The accent is kept for
          the send button, so the most colourful thing on screen is not what the user already
          said. */}
      <div className="rounded-2xl bg-foreground/[0.07] px-4 py-2.5 whitespace-pre-wrap">{text}</div>
      <MessageActions text={text} at={at} end />
    </div>
  );
}

export function AssistantBubble({
  message,
  end,
}: {
  message: Message;
  /** Set on the last thing an answer said: what its actions copy, and its time. See `turnEnds`. */
  end?: TurnEnd;
}) {
  const text = textOf(message);
  return (
    <div className="group/msg max-w-[92%] space-y-1">
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
      {end && <MessageActions text={end.text} at={end.at} />}
    </div>
  );
}
