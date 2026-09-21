import { beforeEach, describe, expect, it, vi } from "vitest";

import { isHidden, loadExpanded, saveExpanded, toggle } from "@/lib/treeState";

/** vitest runs in node, which has no `localStorage`. */
function stubStorage() {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => store.set(k, v),
    removeItem: (k: string) => store.delete(k),
  });
  return store;
}

describe("toggle", () => {
  it("opens a closed directory", () => {
    expect([...toggle(new Set(), "/a")]).toEqual(["/a"]);
  });

  it("closes an open one", () => {
    expect([...toggle(new Set(["/a"]), "/a")]).toEqual([]);
  });

  it("closes what was open inside it", () => {
    // Or reopening springs back to a shape that was collapsed on purpose, and the saved
    // set keeps growing with paths nothing can reach.
    const open = new Set(["/a", "/a/b", "/a/b/c", "/other"]);
    expect([...toggle(open, "/a")].sort()).toEqual(["/other"]);
  });

  it("does not mistake a sibling for a child", () => {
    // `/a/bc` starts with `/a/b`, and only the separator says it is not inside it.
    const open = new Set(["/a/b", "/a/bc"]);
    expect([...toggle(open, "/a/b")]).toEqual(["/a/bc"]);
  });

  it("leaves the set it was given alone", () => {
    const open = new Set(["/a"]);
    toggle(open, "/b");
    expect([...open]).toEqual(["/a"]);
  });
});

describe("loadExpanded and saveExpanded", () => {
  beforeEach(() => stubStorage());

  it("round-trips what was open", () => {
    saveExpanded("/", new Set(["/a", "/a/b"]));
    expect([...loadExpanded("/")].sort()).toEqual(["/a", "/a/b"]);
  });

  it("keeps roots apart", () => {
    saveExpanded("/", new Set(["/a"]));
    saveExpanded("/bucket", new Set(["/bucket/x"]));
    expect([...loadExpanded("/")]).toEqual(["/a"]);
    expect([...loadExpanded("/bucket")]).toEqual(["/bucket/x"]);
  });

  it("is empty for a root nothing was saved for", () => {
    expect([...loadExpanded("/nothing")]).toEqual([]);
  });

  it("is empty rather than broken for a value that is not ours", () => {
    // A tree that starts collapsed is a tree; one built from a half-understood blob is a
    // crash on first paint.
    localStorage.setItem("ailoy.tree:/", "not json");
    expect([...loadExpanded("/")]).toEqual([]);
    localStorage.setItem("ailoy.tree:/", '{"a":1}');
    expect([...loadExpanded("/")]).toEqual([]);
    localStorage.setItem("ailoy.tree:/", '["/a", 7, null]');
    expect([...loadExpanded("/")]).toEqual(["/a"]);
  });

  it("keeps the deepest paths when there are too many", () => {
    const many = new Set<string>();
    for (let i = 0; i < 600; i += 1) many.add(`/shallow${i}`);
    many.add("/a/b/c/d/e/deep");
    saveExpanded("/", many);
    const kept = loadExpanded("/");
    expect(kept.size).toBe(500);
    expect(kept.has("/a/b/c/d/e/deep")).toBe(true);
  });
});

describe("isHidden", () => {
  it("is the leading dot every unix tool reads", () => {
    expect(isHidden(".ssh")).toBe(true);
    expect(isHidden(".DS_Store")).toBe(true);
    expect(isHidden("Documents")).toBe(false);
    // Not a hidden file: the dot is not leading.
    expect(isHidden("archive.tar.gz")).toBe(false);
  });
});
