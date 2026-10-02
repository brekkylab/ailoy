// What a chat waits on: the start's downloads, held in the query cache and moved by the engine.
//
// The engine fetches the console server, the image its VM boots on, and the model list as
// soon as it starts (`core/src/bootstrap.rs`), announces every step as the `bootstrap` event,
// and refuses a run until all of them are done. The composer reads the same status, so the
// send button and the engine never disagree about whether a chat can start.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";

import * as api from "@/api";
import type { BootstrapStatus, BootstrapStep } from "@/types";

export const BOOTSTRAP_KEY = ["bootstrap"] as const;

/**
 * Always asked again on mount: the listener lives with whoever reads the status, so a view
 * that was not on screen for an event catches up by asking.
 */
export const bootstrapQuery = { queryKey: BOOTSTRAP_KEY, queryFn: api.bootstrapStatus, staleTime: 0 };

/** The status, kept current by the engine's `bootstrap` event for as long as it is read. */
export function useBootstrap() {
  const qc = useQueryClient();
  const query = useQuery(bootstrapQuery);
  useEffect(() => {
    // `listen` resolves after a round trip; a StrictMode unmount can land first, and the
    // listener it would have leaked is dropped as soon as it arrives.
    let unlisten: (() => void) | undefined;
    let gone = false;
    void listen<BootstrapStatus>("bootstrap", (e) => qc.setQueryData(BOOTSTRAP_KEY, e.payload)).then((u) => {
      if (gone) u();
      else unlisten = u;
    });
    return () => {
      gone = true;
      unlisten?.();
    };
  }, [qc]);
  return query.data;
}

/** Whether a chat may start. Not until the status has been read: unknown is not ready. */
export function isReady(status: BootstrapStatus | undefined): boolean {
  return status?.ready ?? false;
}

/** Run what has not finished again: the Retry button. */
export function useRetryBootstrap() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.bootstrapRetry,
    onSuccess: (status) => qc.setQueryData(BOOTSTRAP_KEY, status),
  });
}

/** The steps this engine has work for: a skipped one is not shown or counted. */
export function shownSteps(status: BootstrapStatus): BootstrapStep[] {
  return status.steps.filter((s) => s.state !== "skipped");
}

/** How far along it is, over the steps shown. */
export function progress(status: BootstrapStatus): { done: number; total: number } {
  const steps = shownSteps(status);
  return { done: steps.filter((s) => s.state === "done").length, total: steps.length };
}

/** The step that stopped the setup, if one did. */
export function failedStep(status: BootstrapStatus): (BootstrapStep & { state: "failed" }) | undefined {
  return status.steps.find((s): s is BootstrapStep & { state: "failed" } => s.state === "failed");
}

/** How long a step has been running, as a clock reads it: `0:07`, `1:42`, `12:05`. */
export function elapsed(startedAt: number, now: number): string {
  const secs = Math.max(0, Math.floor((now - startedAt) / 1000));
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}

/** The time, every second while `ticking`: what keeps a running step's clock moving. */
export function useNow(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [ticking]);
  return now;
}
