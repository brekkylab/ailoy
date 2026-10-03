// The model catalog's status, held in the query cache and moved by the engine.
//
// The engine refreshes the list on a timer of its own (`core/src/engine.rs`,
// `refresh_catalog`) and announces every change as the `catalog` event. Without listening,
// a window would keep the list it first read — and a first start with no cache reads an
// empty one, so the picker would stay empty until a restart.

import { type QueryClient, useMutation, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { useEffect } from "react";

import * as api from "@/api";
import type { CatalogStatus } from "@/types";

export const CATALOG_KEY = ["catalog"] as const;

export const catalogQuery = { queryKey: CATALOG_KEY, queryFn: api.catalogStatus };

/**
 * Whether the list itself moved, rather than only whether a fetch is in flight.
 *
 * A fetch announces itself twice — starting and ending — and only an ending that brought a
 * new list is worth asking the engine for every model again.
 */
export function listChanged(prev: CatalogStatus | undefined, next: CatalogStatus): boolean {
  return !prev || prev.fetched_at !== next.fetched_at || prev.models !== next.models;
}

/**
 * What the engine answers from the list, and so what is asked for again when it moves:
 * `models` is the list itself, `settings` carries the Bedrock routings read off it, and a
 * session's `usage` takes its context window and prices from it — read during a first start's
 * empty list, the usage bar would otherwise show neither until the session was reopened.
 */
export const DERIVED_FROM_CATALOG = [["models"], ["settings"], ["usage"]] as const;

/** Take a status the engine sent. */
export function applyCatalogStatus(qc: QueryClient, next: CatalogStatus) {
  const prev = qc.getQueryData<CatalogStatus>(CATALOG_KEY);
  qc.setQueryData(CATALOG_KEY, next);
  if (listChanged(prev, next)) {
    for (const queryKey of DERIVED_FROM_CATALOG) void qc.invalidateQueries({ queryKey });
  }
}

/** Listen for the engine's `catalog` event for as long as the window is up. */
export function useCatalogEvents(qc: QueryClient) {
  useEffect(() => {
    // `listen` resolves after a round trip; a StrictMode unmount can land first, and the
    // listener it would have leaked is dropped as soon as it arrives.
    let unlisten: (() => void) | undefined;
    let gone = false;
    void listen<CatalogStatus>("catalog", (e) => applyCatalogStatus(qc, e.payload)).then((u) => {
      if (gone) u();
      else unlisten = u;
    });
    return () => {
      gone = true;
      unlisten?.();
    };
  }, [qc]);
}

/** Fetch the list now: the Retry and Refresh buttons. */
export function useRefreshModels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.modelsRefresh,
    onSuccess: (status) => applyCatalogStatus(qc, status),
  });
}
