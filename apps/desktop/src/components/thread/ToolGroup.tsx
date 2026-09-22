// A run of tool calls, as one closed line.
//
// Everything behind this line is addressed to the agent and not to the reader: the
// arguments a model filled in, and the page of JSON that came back. What the reader needs
// while it happens is *which* tool is running, and what they need afterwards is that it
// happened at all — both of which fit on the closed row. Going looking is a click, and
// then the whole of it is there rather than a summary of it.
//
// Which calls belong together is `lib/thread`'s question, and it answers it from the
// stored list alone. This component only draws what it is handed.

import { Ban, CheckCircle2, ChevronRight, Loader2, XCircle } from "lucide-react";
import { useState } from "react";

import { ToolCallCard } from "@/components/thread/ToolCallCard";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { groupDuration, namedTools, summarizeGroup, type ResolvedCall } from "@/lib/thread";
import { summarizeToolCall } from "@/lib/toolCall";
import { S } from "@/strings";

/** A resolved call, as the card that draws one takes it. */
function Card({ call }: { call: ResolvedCall }) {
  return (
    <ToolCallCard
      name={call.name}
      args={call.args}
      status={call.status}
      result={call.result}
      startedAt={call.startedAt}
      finishedAt={call.finishedAt}
    />
  );
}

export function ToolGroup({
  calls,
  thinking,
  live,
}: {
  calls: ResolvedCall[];
  /** The working the model did between the calls, in the order it did it. */
  thinking: string[];
  /**
   * Whether *this group* is the one the live run is inside — not merely whether some run
   * is going. See `liveGroupKey`: a stopped turn leaves calls nothing ever answered, and
   * a call with no answer means something different in each case.
   */
  live: boolean;
}) {
  /**
   * Open while the run is inside it, and then only because the reader says so.
   *
   * `null` is "the reader has not said", which is the state a group spends its life in: it
   * follows the run in, and closes behind it, which is the whole point of grouping. One
   * click pins it either way for as long as the group is on screen — the keyed segment
   * above keys a group on its first call, so this component survives the group growing and
   * keeps what the reader did to it.
   */
  const [choice, setChoice] = useState<boolean | null>(null);
  const open = choice ?? live;

  // A lone call already reads as one line, with its own icon, name and argument on it.
  // Wrapping it would be a box around a box and a second click to reach anything, so the
  // group starts at two — or at one that has a trace to carry, which has nowhere else to go.
  if (calls.length === 1 && !thinking.length) {
    return <Card call={calls[0]} />;
  }

  const { active, queued, errors, interrupted } = summarizeGroup(calls);
  const took = active ? null : groupDuration(calls);
  const Icon = active ? Loader2 : errors ? XCircle : interrupted ? Ban : CheckCircle2;
  return (
    <Collapsible open={open} onOpenChange={(o) => setChoice(o)} className="my-1 rounded-md border bg-muted/40 text-sm">
      {/* Base UI marks an open trigger with `data-panel-open`, not Radix's `data-state`. */}
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left">
        <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
        {/* One icon, in the slot a call's own card keeps its status in, so a group row and
            the rows inside it read as the same kind of line. What says this one is a group
            is the count beside it. */}
        <Icon
          className={
            active
              ? "size-4 shrink-0 animate-spin"
              : errors
                ? "size-4 shrink-0 text-destructive"
                : "size-4 shrink-0 text-muted-foreground"
          }
        />
        {active ? (
          // What is happening, not what has happened: the call the agent is inside, drawn
          // the way its own card draws it, and the count of the ones running beside it.
          <>
            <span className="shrink-0 font-mono text-xs text-muted-foreground">{active.name}</span>
            <span className="flex-1 truncate font-mono text-xs">{summarizeToolCall(active.name, active.args)}</span>
            {queued > 0 && (
              <span className="shrink-0 text-xs text-muted-foreground">
                +{queued} {S.queued}
              </span>
            )}
          </>
        ) : (
          <>
            <span className="shrink-0 text-xs">
              {calls.length} {S.toolCalls}
            </span>
            {/* Which tools, after how many — the part the row drops first when it is
                narrow, because the count is what the row is for. */}
            <span className="flex-1 truncate font-mono text-xs text-muted-foreground">{namedTools(calls)}</span>
          </>
        )}
        {/* Why a group of finished calls is not simply done, which the row otherwise
            gives no sign of. Both can be true of one group. */}
        {errors > 0 && (
          <span className="shrink-0 text-xs text-destructive">
            {errors} {S.callsFailed}
          </span>
        )}
        {interrupted > 0 && (
          <span className="shrink-0 text-xs text-muted-foreground">
            {interrupted} {S.callsInterrupted}
          </span>
        )}
        {took != null && <span className="shrink-0 text-xs text-muted-foreground">{took}s</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-1 border-t bg-background/40 px-2 py-2">
        {/* The working the model did on its way through the group. Above the calls because
            that is where it happened, and inside the group because it is addressed to the
            agent — the same reason everything else here is behind that line. */}
        {thinking.map((thought, i) => (
          <pre
            key={i}
            className="max-h-32 overflow-auto rounded bg-muted/40 p-2 text-xs whitespace-pre-wrap text-muted-foreground"
          >
            {thought}
          </pre>
        ))}
        {/* Keyed on the position as well as the id: a group gathers the calls of several
            messages, so an id that repeats — or an empty one — is a duplicate key across a
            whole run rather than within one message. The position is unique within a group
            and never moves, because calls are only ever appended. */}
        {calls.map((c, i) => (
          <Card key={`${c.id}#${i}`} call={c} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
