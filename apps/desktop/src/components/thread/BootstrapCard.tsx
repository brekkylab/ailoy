// The start's downloads, above the composer until a chat can run.
//
// One row per step the engine has work for — the console server, the image its VM boots
// on, the model list — each with what it is doing and, while it runs, for how long: none of
// the three reports a byte count, so a moving clock is what says it has not stalled. A
// failure stops its chain (the image waits on the server) and puts Retry under the list.

import { CheckCircle2, Circle, Loader2, TriangleAlert, XCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { elapsed, failedStep, progress, shownSteps, useNow, useRetryBootstrap } from "@/lib/bootstrap";
import { S } from "@/strings";
import type { BootstrapStatus, BootstrapStep } from "@/types";

function StepIcon({ step }: { step: BootstrapStep }) {
  switch (step.state) {
    case "running":
      return <Loader2 className="size-4 shrink-0 animate-spin text-primary" />;
    case "done":
      return <CheckCircle2 className="size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />;
    case "failed":
      return <XCircle className="size-4 shrink-0 text-destructive" />;
    default:
      return <Circle className="size-4 shrink-0 text-muted-foreground/50" />;
  }
}

function StepRow({ step, now }: { step: BootstrapStep; now: number }) {
  const text = S.bootstrapSteps[step.id];
  return (
    <li className="flex items-start gap-2.5 py-1.5" data-state={step.state}>
      <span className="mt-0.5">
        <StepIcon step={step} />
      </span>
      <div className="min-w-0 flex-1">
        <div className={step.state === "pending" ? "text-sm text-muted-foreground" : "text-sm"}>{text.label}</div>
        {step.state === "failed" ? (
          <div className="truncate text-xs text-destructive" title={step.message}>
            {step.message}
          </div>
        ) : (
          step.state === "running" && <div className="text-xs text-muted-foreground">{text.detail}…</div>
        )}
      </div>
      <span className="shrink-0 pt-0.5 text-xs tabular-nums text-muted-foreground">
        {step.state === "running"
          ? elapsed(step.started_at, now)
          : step.state === "done"
            ? S.bootstrapDone
            : step.state === "pending"
              ? S.bootstrapPending
              : null}
      </span>
    </li>
  );
}

export function BootstrapCard({ status }: { status: BootstrapStatus }) {
  const retry = useRetryBootstrap();
  const steps = shownSteps(status);
  const now = useNow(steps.some((s) => s.state === "running"));
  const failed = failedStep(status);
  const { done, total } = progress(status);

  return (
    <section
      aria-label={S.bootstrapTitle}
      aria-live="polite"
      className="mb-2 rounded-2xl border bg-card px-4 pt-3 pb-2.5 shadow-sm"
    >
      <div className="flex items-center gap-2">
        {failed ? (
          <TriangleAlert className="size-4 shrink-0 text-destructive" />
        ) : (
          <Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" />
        )}
        <h2 className="text-sm font-medium">{failed ? S.bootstrapFailedTitle : S.bootstrapTitle}</h2>
        <span className="ml-auto text-xs tabular-nums text-muted-foreground">
          {done} / {total}
        </span>
      </div>
      <Progress value={total ? (done / total) * 100 : 0} className="mt-2.5" aria-label={S.bootstrapTitle} />
      <ul className="mt-1.5">
        {steps.map((step) => (
          <StepRow key={step.id} step={step} now={now} />
        ))}
      </ul>
      <div className="mt-1 flex items-center gap-3 border-t pt-2">
        <p className="flex-1 text-xs text-muted-foreground">{failed ? S.bootstrapFailedHint : S.bootstrapHint}</p>
        {failed && (
          <Button size="sm" variant="outline" onClick={() => retry.mutate()} disabled={retry.isPending}>
            {S.retry}
          </Button>
        )}
      </div>
    </section>
  );
}
