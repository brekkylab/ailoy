// The one line above the composer: how full the context window is, what the session has
// spent, and how much provider headroom is left.
//
// Two sources feed it. `session_usage` is the persisted total the engine recomputes from
// stored messages; a run's `usage` event carries the numbers of the model message that
// just completed and replaces — never sums — the live totals. While a run is in flight
// the live numbers are the fresher ones, so they win; the persisted total fills in
// before the first event of a session and after a reload. Only a run's terminal event
// moves the persisted total, so a transition into `done`/`cancelled`/`error` is what
// refetches it — nothing else would.

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import * as api from "@/api";
import { Progress } from "@/components/ui/progress";
import { type LiveRun, selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";
import type { RateLimitWindow } from "@/types";

const fmt = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);

const isTerminal = (s: LiveRun["status"]) => s === "done" || s === "cancelled" || s === "error";

/**
 * Whether a window carries the two numbers a percentage needs. A provider that sends the
 * header but leaves a window empty must not put a bare "분당 잔여" label on the bar with
 * nothing after it, so the group asks this of all four before it renders at all.
 */
const hasWindow = (
  w: RateLimitWindow | null | undefined,
): w is RateLimitWindow & { limit: number; remaining: number } =>
  w != null && w.limit != null && w.remaining != null && w.limit !== 0;

/** One provider window. Rendered only when the provider sent both numbers. */
function Window({ label, w }: { label: string; w: RateLimitWindow | null | undefined }) {
  if (!hasWindow(w)) return null;
  const pct = Math.round((w.remaining / w.limit) * 100);
  const resetIn = w.reset_at_ms != null ? Math.max(0, Math.round((w.reset_at_ms - Date.now()) / 1000)) : null;
  return (
    <span
      title={`${w.remaining}/${w.limit}${resetIn != null ? ` · ${S.resetIn} ${resetIn}s` : ""}`}
      className={pct < 10 ? "text-destructive" : undefined}
    >
      {label} {pct}%
    </span>
  );
}

export function UsageBar({ sessionId }: { sessionId: string }) {
  const qc = useQueryClient();
  const usage = useQuery({ queryKey: ["usage", sessionId], queryFn: () => api.sessionUsage(sessionId) });
  const live = useRunStore(selectRun(sessionId));

  // The engine writes the run's messages — and with them the session's usage row and the
  // session's `updated_at` — as the run ends. Watching the status rather than the event
  // stream keeps this to one refetch per run, and the ref is keyed by session so that
  // switching between sessions does not read the previous one's status as a transition.
  const seen = useRef<{ id: string; status: LiveRun["status"] } | null>(null);
  const status = live.status;
  useEffect(() => {
    const was = seen.current?.id === sessionId ? seen.current.status : null;
    seen.current = { id: sessionId, status };
    if (was == null || was === status || !isTerminal(status)) return;
    void qc.invalidateQueries({ queryKey: ["usage", sessionId] });
    void qc.invalidateQueries({ queryKey: ["sessions"] });
  }, [qc, sessionId, status]);

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

  return (
    <div className="flex items-center gap-4 px-1 pb-1 text-xs text-muted-foreground">
      {pct != null && (
        <div className="flex items-center gap-2" title={`${used} / ${limit}`}>
          <span>{S.contextUsage}</span>
          <Progress value={pct} className="h-1.5 w-28" />
          <span>{pct.toFixed(0)}%</span>
        </div>
      )}
      {u && (
        <span
          title={`in ${u.input_tokens} · out ${u.output_tokens} · cache r ${u.cache_read_tokens} w ${u.cache_write_tokens}`}
        >
          {S.sessionTokens} {fmt(u.input_tokens + u.output_tokens)}
        </span>
      )}
      {u?.estimated_cost_usd != null && (
        <span>
          {S.estimatedCost} ${u.estimated_cost_usd.toFixed(4)}
        </span>
      )}
      {rl && (
        <span className="flex gap-2">
          <span>{S.rateLimit}</span>
          <Window label={S.rlRequests} w={rl.requests} />
          <Window label={S.rlTokens} w={rl.tokens} />
          <Window label={S.rlInput} w={rl.input_tokens} />
          <Window label={S.rlOutput} w={rl.output_tokens} />
        </span>
      )}
    </div>
  );
}
