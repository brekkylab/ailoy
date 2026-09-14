// One tool call, folded shut.
//
// The header is the whole call in a line — icon, tool name, the argument that identifies
// it — because a thread is mostly scrolled past; the arguments and the result only appear
// when the row is opened. Live and stored calls render through the same component, so a
// call does not jump when the run ends and the thread refetches.

import { Ban, CheckCircle2, ChevronRight, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { fieldsOf, summarizeToolCall } from "@/lib/toolCall";
import type { ToolStatus } from "@/store/runs";
import { S } from "@/strings";

const ICONS = { running: Loader2, done: CheckCircle2, error: XCircle, interrupted: Ban } as const;

/**
 * Seconds since a running call began.
 *
 * On its own clock, not on the render clock: a tool the engine is waiting on emits no
 * events, so nothing re-renders the thread while it runs and a duration read during
 * render would freeze at the moment the call was announced — exactly the stretch the
 * number exists to describe.
 */
function useElapsedSeconds(startedAt: number | undefined): number | null {
  // The clock is the state; the duration is derived from it. Storing the duration would
  // mean writing it from inside the effect, which is a render the tick does not need.
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (startedAt == null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  if (startedAt == null) return null;
  // Before the first tick — and for the first second of a call that reused this card —
  // the clock is behind the call, which reads as the zero it is.
  return Math.max(0, Math.round((now - startedAt) / 1000));
}

export function ToolCallCard({
  name,
  args,
  status,
  result,
  startedAt,
}: {
  name: string;
  args: unknown;
  status: ToolStatus;
  result?: unknown;
  /** Only the live thread has one; a call read back from storage does not. */
  startedAt?: number;
}) {
  const Icon = ICONS[status];
  const elapsed = useElapsedSeconds(status === "running" ? startedAt : undefined);
  const fields = result === undefined ? [] : fieldsOf(result);
  return (
    <Collapsible className="my-1 rounded-md border bg-muted/40 text-sm">
      {/* Base UI marks an open trigger with `data-panel-open`, not Radix's `data-state`. */}
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left">
        <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
        <Icon
          className={
            status === "running"
              ? "size-4 shrink-0 animate-spin"
              : status === "error"
                ? "size-4 shrink-0 text-destructive"
                : "size-4 shrink-0"
          }
        />
        <span className="shrink-0 font-mono text-xs text-muted-foreground">{name}</span>
        <span className="flex-1 truncate font-mono text-xs">{summarizeToolCall(name, args)}</span>
        {status === "running" && (
          <span className="shrink-0 text-xs text-muted-foreground">
            {S.running}
            {elapsed ? ` · ${elapsed}s` : ""}
          </span>
        )}
        {status === "interrupted" && <span className="shrink-0 text-xs text-muted-foreground">{S.interrupted}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 border-t px-3 py-2">
        <div className="font-mono text-xs text-muted-foreground">{S.arguments}</div>
        <pre className="max-h-40 overflow-auto rounded bg-background p-2 font-mono text-xs">
          {JSON.stringify(args, null, 2)}
        </pre>
        {fields.map((f) =>
          f.kind === "inline" ? (
            <div key={f.key} className="font-mono text-xs">
              <span className="text-muted-foreground">{f.key}: </span>
              {f.value}
            </div>
          ) : (
            <div key={f.key || "_"}>
              {f.key && <div className="font-mono text-xs text-muted-foreground">{f.key}</div>}
              <pre className="max-h-64 overflow-auto rounded bg-background p-2 font-mono text-xs whitespace-pre-wrap">
                {f.value}
              </pre>
            </div>
          ),
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
