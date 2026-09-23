// The context ring in the composer: how full the context window is, and — a click away, in
// a dialog — what the session has spent and how much provider headroom is left.
//
// Two sources feed it. `session_usage` is the persisted total the engine recomputes from
// stored messages; a run's `usage` event carries the numbers of the model message that
// just completed and replaces — never sums — the live totals. While a run is in flight
// the live numbers are the fresher ones, so they win; the persisted total fills in
// before the first event of a session and after a reload. Only a run's terminal event
// moves the persisted total, so a transition into `done`/`cancelled`/`error` is what
// refetches it — nothing else would.
//
// That same transition is the moment *everything* a run changed becomes readable, and
// this is the one component mounted per session that watches for it, so the whole
// end-of-run refetch lives here rather than being spread over four components that would
// each have to re-derive the transition. See the effect below.

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "cn";
import { useState } from "react";

import * as api from "@/api";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { useRunStore, useRunTerminal, selectRun } from "@/store/runs";
import { formatTokens } from "@/lib/tokens";
import { changedByRun } from "@/lib/queryClient";
import { S } from "@/strings";
import type { RateLimitWindow } from "@/types";

/**
 * Whether a window carries the two numbers a percentage needs. A provider that sends the
 * header but leaves a window empty must not put a bare "Rate limit" label on the bar with
 * nothing after it, so the group asks this of all four before it renders at all.
 */
const hasWindow = (
  w: RateLimitWindow | null | undefined,
): w is RateLimitWindow & { limit: number; remaining: number } =>
  w != null && w.limit != null && w.remaining != null && w.limit !== 0;

/** A bar of how much of something there is: the track in the muted tone, the fill in `tone`. */
function Meter({ pct, tone = "bg-foreground/70" }: { pct: number; tone?: string }) {
  return (
    <div className="h-1.5 overflow-hidden rounded-full bg-muted">
      <div className={cn("h-full rounded-full transition-[width]", tone)} style={{ width: `${Math.min(100, pct)}%` }} />
    </div>
  );
}

/**
 * One provider window, as a row of the dialog. Rendered only when the provider sent both
 * numbers. A token window reads in k/M like every other token figure here; the requests
 * window is a count of calls, which stays exact.
 */
function WindowRow({
  label,
  w,
  tokens = true,
}: {
  label: string;
  w: RateLimitWindow | null | undefined;
  tokens?: boolean;
}) {
  if (!hasWindow(w)) return null;
  const pct = (w.remaining / w.limit) * 100;
  const resetIn = w.reset_at_ms != null ? Math.max(0, Math.round((w.reset_at_ms - Date.now()) / 1000)) : null;
  const low = pct < 10;
  const num = (n: number) => (tokens ? formatTokens(n) : n.toLocaleString());
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className={cn("font-medium", low && "text-destructive")}>{label}</span>
        <span className="text-muted-foreground tabular-nums">
          {num(w.remaining)} / {num(w.limit)} {S.remaining}
          {resetIn != null && ` · ${S.resetIn} ${resetIn}s`}
        </span>
      </div>
      <Meter pct={pct} tone={low ? "bg-destructive" : undefined} />
    </div>
  );
}

/** A labelled figure in the token grid, in k/M like the rest of the dialog — exact below 1k. */
function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="tabular-nums">{formatTokens(value)}</dd>
    </div>
  );
}

/** A section heading in the dialog. */
function Heading({ children }: { children: React.ReactNode }) {
  return <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{children}</h3>;
}

export function UsageBar({ sessionId }: { sessionId: string }) {
  const qc = useQueryClient();
  const usage = useQuery({ queryKey: ["usage", sessionId], queryFn: () => api.sessionUsage(sessionId) });
  const [open, setOpen] = useState(false);
  const live = useRunStore(selectRun(sessionId));

  // The engine writes the run's messages — and with them the session's usage row and the
  // session's `updated_at` — as the run ends. Watching the status rather than the event
  // stream keeps this to one refetch per run.
  //
  // The files are here for a different reason: the agent writes through the engine, not
  // through this app's own mutations, so nothing in `WorkspacePanel` or `FileTree` ever
  // hears about it and the tree sits on a listing from before the run. Only where it
  // writes, though — see `changedByRun`, which is what keeps a run from re-reading a whole
  // Notion tree the agent cannot even write to.
  useRunTerminal(sessionId, () => {
    void qc.invalidateQueries({ queryKey: ["usage", sessionId] });
    void qc.invalidateQueries({ queryKey: ["sessions"] });
    void qc.invalidateQueries({ predicate: (q) => changedByRun(q.queryKey) });
  });

  const used = live.contextUsed ?? usage.data?.context_used ?? null;
  const limit = live.contextLimit ?? usage.data?.context_limit ?? null;
  const pct = used != null && limit ? Math.min(100, (used / limit) * 100) : null;
  const u = usage.data;
  // Anthropic sends all four windows; another provider may send the header with none of
  // them filled in. Nothing to show is nothing to label.
  const r = live.rateLimit;
  const rl =
    r && (hasWindow(r.requests) || hasWindow(r.tokens) || hasWindow(r.input_tokens) || hasWindow(r.output_tokens))
      ? r
      : null;

  // Said on the bar itself, not only in the tooltip: a provider this close to its limit is
  // about to fail the next call, which is worth the room.
  const low = rl
    ? [rl.requests, rl.tokens, rl.input_tokens, rl.output_tokens].some(
        (w) => hasWindow(w) && w.remaining / w.limit < 0.1,
      )
    : false;
  if (pct == null && !u) return null;

  // A ring and a number beside the send button, where Claude and Codex put it, rather than
  // a line of five figures above the box. The figures are one click away, in a dialog with
  // the room to lay them out — a tooltip is gone as soon as the pointer moves, which is not
  // how anyone reads a cost.
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-label={S.usageTitle}
        title={S.usageTitle}
        className={cn(
          "flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs tabular-nums text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
          low && "text-destructive hover:text-destructive",
        )}
      >
        {pct != null && <ContextRing pct={pct} />}
        {pct != null ? `${pct.toFixed(0)}%` : u && formatTokens(u.input_tokens + u.output_tokens)}
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="gap-6 p-5 sm:max-w-md">
          <DialogTitle>{S.usageTitle}</DialogTitle>
          {pct != null && used != null && limit != null && (
            <section className="space-y-2">
              <div className="flex items-baseline justify-between">
                <Heading>{S.contextWindow}</Heading>
                <span className="text-2xl font-semibold tabular-nums">{pct.toFixed(pct < 10 ? 1 : 0)}%</span>
              </div>
              <Meter pct={pct} tone={pct > 90 ? "bg-destructive" : undefined} />
              <p className="text-xs text-muted-foreground tabular-nums">
                {formatTokens(used)} {S.ofTokens} {formatTokens(limit)} {S.tokensUsed}
              </p>
            </section>
          )}
          {u && (
            <section className="space-y-2">
              <div className="flex items-baseline justify-between">
                <Heading>{S.sessionTokens}</Heading>
                <span className="text-sm font-medium tabular-nums">
                  {formatTokens(u.input_tokens + u.output_tokens)}
                </span>
              </div>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm">
                <Stat label={S.inputTokens} value={u.input_tokens} />
                <Stat label={S.outputTokens} value={u.output_tokens} />
                <Stat label={S.cacheRead} value={u.cache_read_tokens} />
                <Stat label={S.cacheWrite} value={u.cache_write_tokens} />
              </dl>
            </section>
          )}
          {u?.estimated_cost_usd != null && (
            <section className="space-y-1">
              <div className="flex items-baseline justify-between">
                <Heading>{S.estimatedCost}</Heading>
                <span className="text-2xl font-semibold tabular-nums">${u.estimated_cost_usd.toFixed(4)}</span>
              </div>
              <p className="text-xs text-muted-foreground">{S.costNote}</p>
            </section>
          )}
          {rl && (
            <section className="space-y-3">
              <Heading>{S.rateLimit}</Heading>
              <WindowRow label={S.rlRequests} w={rl.requests} tokens={false} />
              <WindowRow label={S.rlTokens} w={rl.tokens} />
              <WindowRow label={S.rlInput} w={rl.input_tokens} />
              <WindowRow label={S.rlOutput} w={rl.output_tokens} />
            </section>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

/** How full the context window is, as a ring: the track in the muted tone, the used arc in the text's. */
function ContextRing({ pct }: { pct: number }) {
  const r = 5.5;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 14 14" className="size-3.5 -rotate-90" aria-hidden>
      <circle cx="7" cy="7" r={r} fill="none" strokeWidth="2" className="stroke-foreground/15" />
      <circle
        cx="7"
        cy="7"
        r={r}
        fill="none"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray={`${(Math.max(pct, 2) / 100) * c} ${c}`}
        className="stroke-current"
      />
    </svg>
  );
}
