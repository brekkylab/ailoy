import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import type { CatalogStatus } from "@/types";

import { applyCatalogStatus, CATALOG_KEY, listChanged } from "./catalog";

const status = (over: Partial<CatalogStatus> = {}): CatalogStatus => ({
  fetched_at: 1_000,
  models: 10,
  refreshing: false,
  error: null,
  ...over,
});

describe("the catalog status", () => {
  it("counts a new list as a change, and a fetch starting or failing as none", () => {
    expect(listChanged(undefined, status())).toBe(true);
    expect(listChanged(status(), status({ refreshing: true }))).toBe(false);
    expect(listChanged(status(), status({ error: "offline" }))).toBe(false);
    expect(listChanged(status(), status({ fetched_at: 2_000 }))).toBe(true);
    // A first fetch landing: the empty list becomes one.
    expect(listChanged(status({ fetched_at: null, models: 0 }), status())).toBe(true);
  });

  it("asks again for what is read off the list only when the list moved", () => {
    const qc = new QueryClient();
    qc.setQueryData(CATALOG_KEY, status());
    qc.setQueryData(["models"], []);
    qc.setQueryData(["settings"], {});
    // Every session's usage, whichever is open: each takes its prices from the list.
    qc.setQueryData(["usage", "s1"], {});
    qc.setQueryData(["sessions"], []);
    const stale = (key: unknown[]) => qc.getQueryState(key)?.isInvalidated;

    applyCatalogStatus(qc, status({ refreshing: true }));
    expect(qc.getQueryData(CATALOG_KEY)).toEqual(status({ refreshing: true }));
    expect(stale(["models"])).toBe(false);

    applyCatalogStatus(qc, status({ fetched_at: 2_000 }));
    expect([stale(["models"]), stale(["settings"]), stale(["usage", "s1"])]).toEqual([true, true, true]);
    expect(stale(["sessions"])).toBe(false);
  });
});
