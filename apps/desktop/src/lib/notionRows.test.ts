import { beforeEach, describe, expect, it, vi } from "vitest";

import { recalledRow, rememberRow } from "./notionRows";

/** vitest runs in node, which has no `localStorage`. */
function fakeStorage() {
  const held = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => held.get(k) ?? null,
    setItem: (k: string, v: string) => void held.set(k, v),
    removeItem: (k: string) => void held.delete(k),
  });
  return held;
}

beforeEach(() => {
  vi.unstubAllGlobals();
  fakeStorage();
});

const ROW = "/notion/pages/A__1a2b";

describe("notion row memory", () => {
  it("gives back what a row's json said", () => {
    rememberRow(ROW, { icon: "📝", leaf: false, title: "A page with spaces" });
    expect(recalledRow(ROW)).toEqual({ icon: "📝", leaf: false, title: "A page with spaces" });
    rememberRow(ROW, { icon: null, leaf: true, title: null });
    expect(recalledRow(ROW)).toEqual({ icon: null, leaf: true, title: null });
  });

  it("knows nothing about a row it has not seen", () => {
    expect(recalledRow(ROW)).toBeNull();
  });

  it("keeps the rows most recently seen", () => {
    for (let i = 0; i < 1005; i += 1) rememberRow(`/notion/pages/p${i}`, { icon: null, leaf: true, title: null });
    expect(recalledRow("/notion/pages/p0")).toBeNull();
    expect(recalledRow("/notion/pages/p1004")).toEqual({ icon: null, leaf: true, title: null });

    // Seeing a row again moves it back to the end, so a reader's own pages are not the ones
    // the cap drops.
    rememberRow("/notion/pages/p5", { icon: "📌", leaf: false, title: "Pinned" });
    for (let i = 0; i < 999; i += 1) rememberRow(`/notion/pages/q${i}`, { icon: null, leaf: true, title: null });
    expect(recalledRow("/notion/pages/p5")).toEqual({ icon: "📌", leaf: false, title: "Pinned" });
  });

  it("reads anything it does not understand as nothing remembered", () => {
    localStorage.setItem("ailoy.notionRows", "not json");
    expect(recalledRow(ROW)).toBeNull();
    localStorage.setItem("ailoy.notionRows", JSON.stringify({ [ROW]: "wrong shape" }));
    expect(recalledRow(ROW)).toBeNull();
    localStorage.setItem("ailoy.notionRows", JSON.stringify({ [ROW]: [1, 2] }));
    expect(recalledRow(ROW)).toBeNull();

    // A blob from before rows remembered their titles still answers for what it has.
    localStorage.setItem("ailoy.notionRows", JSON.stringify({ [ROW]: ["📝", true] }));
    expect(recalledRow(ROW)).toEqual({ icon: "📝", leaf: true, title: null });
  });

  it("draws a row even where there is no storage at all", () => {
    vi.unstubAllGlobals();
    expect(() => rememberRow(ROW, { icon: "📝", leaf: true, title: null })).not.toThrow();
    expect(recalledRow(ROW)).toBeNull();
  });
});
