import { describe, expect, it, vi } from "vitest";

import "@/lib/mapUpsert";

// Node has neither method either, so these exercise the polyfill itself rather than the
// engine's — which is the point: the environment that needs it cannot run these tests.
interface Upsert {
  getOrInsert(key: unknown, value: unknown): unknown;
  getOrInsertComputed(key: unknown, callback: (key: unknown) => unknown): unknown;
}
const upsert = (m: Map<unknown, unknown>) => m as unknown as Upsert;

describe("getOrInsert", () => {
  it("inserts a missing key and returns what it inserted", () => {
    const m = new Map<string, number>();
    expect(upsert(m).getOrInsert("a", 1)).toBe(1);
    expect(m.get("a")).toBe(1);
  });

  it("leaves an existing key alone", () => {
    const m = new Map([["a", 1]]);
    expect(upsert(m).getOrInsert("a", 2)).toBe(1);
    expect(m.get("a")).toBe(1);
  });
});

describe("getOrInsertComputed", () => {
  it("computes once for a missing key", () => {
    const m = new Map<string, number>();
    const make = vi.fn(() => 7);
    expect(upsert(m).getOrInsertComputed("a", make)).toBe(7);
    expect(upsert(m).getOrInsertComputed("a", make)).toBe(7);
    expect(make).toHaveBeenCalledTimes(1);
  });

  it("passes the key to the callback", () => {
    const m = new Map<string, string>();
    expect(upsert(m).getOrInsertComputed("k", (k) => `${String(k)}!`)).toBe("k!");
  });

  it("treats a stored undefined as present, and does not recompute it", () => {
    // The case a null check would get wrong: pdf.js caches promises in these maps, and
    // calling the callback twice would start the same worker request twice.
    const m = new Map([["a", undefined]]);
    const make = vi.fn(() => 1);
    expect(upsert(m).getOrInsertComputed("a", make)).toBeUndefined();
    expect(make).not.toHaveBeenCalled();
  });
});
