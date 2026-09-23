// One tool call, as the line a reader scans past.
//
// A thread is mostly scrolled past, so the closed row is the call in words — `Ran`
// `ls -la workspace`, `Read` `~/notes.md` — with the long absolute paths the agent has to
// use written the way the user knows them (`lib/toolCall`'s `shortenPaths`). Opened, a
// command reads as a terminal would show it: the command after a `$`, what it printed, and
// its exit code only when that is news. The arguments and the result as the tools returned
// them are one more click away, behind Raw, for when the tidy version is not the question.
//
// Live and stored calls render through the same component, so a call does not jump when
// the run ends and the thread refetches.

import { Ban, CheckCircle2, ChevronRight, Loader2, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { ROW } from "@/components/thread/row";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  describeCall,
  fieldsOf,
  isErrorResult,
  previewLines,
  shellResult,
  shortenPaths,
  summarizeToolCall,
  type PathRoot,
} from "@/lib/toolCall";
import { usePathRoots } from "@/lib/usePathRoots";
import type { ToolStatus } from "@/store/runs";
import { S } from "@/strings";

const ICONS = { running: Loader2, done: CheckCircle2, error: XCircle, interrupted: Ban } as const;

/** The status mark every call row and group row starts with, in the same slot. */
export function StatusIcon({ status }: { status: ToolStatus }) {
  const Icon = ICONS[status];
  return (
    <Icon
      className={
        status === "running"
          ? "size-3.5 shrink-0 animate-spin"
          : status === "error"
            ? "size-3.5 shrink-0 text-destructive"
            : "size-3.5 shrink-0"
      }
    />
  );
}

/** The chevron at a row's end: there when the row is hovered or open, not on every line. */
export function RowChevron() {
  return (
    <ChevronRight className="size-3.5 shrink-0 opacity-0 transition group-hover:opacity-100 group-data-[panel-open]:rotate-90 group-data-[panel-open]:opacity-100" />
  );
}

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
  // Seeded at mount rather than at 0, so the first second reads `0s` instead of a blank
  // that fills in a beat later.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt == null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  if (startedAt == null) return null;
  // A card reused by a later call can hold a clock older than the call for up to a tick.
  return Math.max(0, Math.round((now - startedAt) / 1000));
}

export function ToolCallCard({
  name,
  args,
  status,
  result,
  startedAt,
  finishedAt,
  preparing,
}: {
  name: string;
  args: unknown;
  status: ToolStatus;
  /** Still being written by the model: `args` is a preview, and there is no result yet. */
  preparing?: boolean;
  result?: unknown;
  /** When the call began — the live run's clock, or the stored result row's `started_at`. */
  startedAt?: number;
  /** When it ended — the live run saw it, or the stored result row was written. Absent while running. */
  finishedAt?: number;
}) {
  const elapsed = useElapsedSeconds(status === "running" ? startedAt : undefined);
  // How long the call actually took, once it is over — from when the model began writing it,
  // not only the running, see `tool_call_preparing` in the run store. The same number after
  // a reload: the engine keeps when a call began on its result row. A row from before it did
  // has no start, and the card says nothing rather than guess from the message's own clock.
  const took =
    status !== "running" && startedAt != null && finishedAt != null
      ? Math.max(0, Math.round((finishedAt - startedAt) / 1000))
      : null;
  const roots = usePathRoots();
  const line = describeCall(name, args, roots, status === "running");
  return (
    <Collapsible className="text-sm">
      {/* Base UI marks an open trigger with `data-panel-open`, not Radix's `data-state`. */}
      <CollapsibleTrigger className={ROW}>
        <StatusIcon status={status} />
        <span className="shrink-0">{line.verb}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-foreground/80">
          {line.target || (preparing ? "…" : "")}
        </span>
        {/* The verb already says it is running; what is left to say is for how long. */}
        {status === "running" && elapsed != null && (
          <span className="shrink-0 text-xs tabular-nums">{elapsed}s</span>
        )}
        {status === "interrupted" && (
          <span className="shrink-0 text-xs">
            {S.interrupted}
            {took == null ? "" : ` · ${took}s`}
          </span>
        )}
        {/* A finished call keeps its number where the running one had it, so the row does
            not reflow the instant the result lands. */}
        {(status === "done" || status === "error") && took != null && (
          <span className="shrink-0 text-xs tabular-nums">{took}s</span>
        )}
        <RowChevron />
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1 mb-2 ml-7">
        <CallDetail name={name} args={args} result={result} roots={roots} />
      </CollapsibleContent>
    </Collapsible>
  );
}

/** Some output, its first lines at a time: a `ls -R` should not be a page of scroll. */
function Output({ text, tone }: { text: string; tone?: string }) {
  const [all, setAll] = useState(false);
  if (!text.trim()) return null;
  const { shown, hidden } = previewLines(text);
  return (
    <div className="border-t px-3 py-2">
      <pre className={`max-h-96 overflow-auto whitespace-pre-wrap break-words ${tone ?? ""}`}>
        {all ? text.replace(/\n+$/, "") : shown}
      </pre>
      {hidden > 0 && !all && (
        <button className="mt-1 text-[11px] text-muted-foreground hover:text-foreground" onClick={() => setAll(true)}>
          {S.showMore} ({hidden} {S.lines})
        </button>
      )}
    </div>
  );
}

/** What an opened call shows: the call as run, and what came back. */
function CallDetail({
  name,
  args,
  result,
  roots,
}: {
  name: string;
  args: unknown;
  result: unknown;
  roots: PathRoot[];
}) {
  const [raw, setRaw] = useState(false);
  const full = shortenPaths(summarizeToolCall(name, args), roots);
  const shell = name === "shell" ? shellResult(result) : null;
  return (
    <div className="overflow-hidden rounded-md border bg-muted/40 font-mono text-xs">
      {raw ? (
        <RawView args={args} result={result} />
      ) : (
        <>
          <pre className="px-3 py-2 break-words whitespace-pre-wrap">
            {name === "shell" && <span className="text-muted-foreground select-none">$ </span>}
            {full}
          </pre>
          {shell ? (
            <>
              <Output text={shell.stdout} />
              <Output text={shell.stderr} tone="text-destructive" />
              {/* Only what is news: a zero exit is every command that worked. */}
              {(shell.exitCode !== 0 && shell.exitCode != null) || shell.timedOut || shell.truncated ? (
                <div className="flex gap-3 border-t px-3 py-1.5 text-[11px] text-muted-foreground">
                  {shell.exitCode !== 0 && shell.exitCode != null && (
                    <span className="text-destructive">
                      {S.exitCode} {shell.exitCode}
                    </span>
                  )}
                  {shell.timedOut && <span className="text-destructive">{S.timedOut}</span>}
                  {shell.truncated && <span>{S.outputTruncated}</span>}
                </div>
              ) : null}
            </>
          ) : result === undefined ? null : typeof result === "string" ? (
            <Output text={result} />
          ) : (
            <div className={`space-y-1 border-t px-3 py-2 ${isErrorResult(result) ? "text-destructive" : ""}`}>
              {fieldsOf(result).map((f) =>
                f.kind === "inline" ? (
                  <div key={f.key}>
                    <span className="text-muted-foreground">{f.key}: </span>
                    {f.value}
                  </div>
                ) : (
                  <div key={f.key || "_"}>
                    {f.key && <div className="text-muted-foreground">{f.key}</div>}
                    <pre className="max-h-64 overflow-auto whitespace-pre-wrap">{f.value}</pre>
                  </div>
                ),
              )}
            </div>
          )}
        </>
      )}
      <div className="flex justify-end border-t px-2 py-1">
        <button
          className="font-sans text-[11px] text-muted-foreground hover:text-foreground"
          onClick={() => setRaw((r) => !r)}
          aria-pressed={raw}
        >
          {raw ? S.formatted : S.raw}
        </button>
      </div>
    </div>
  );
}

/** The call as the model sent it and the result as the tool returned it, untouched. */
function RawView({ args, result }: { args: unknown; result: unknown }) {
  return (
    <div className="space-y-2 px-3 py-2">
      <div className="text-muted-foreground">{S.arguments}</div>
      <pre className="max-h-40 overflow-auto whitespace-pre-wrap">{JSON.stringify(args, null, 2)}</pre>
      {result !== undefined && (
        <>
          <div className="text-muted-foreground">{S.result}</div>
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap">
            {typeof result === "string" ? result : JSON.stringify(result, null, 2)}
          </pre>
        </>
      )}
    </div>
  );
}
