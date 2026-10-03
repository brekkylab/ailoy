import { describe, expect, it } from "vitest";

import { BYTES_GC_MS, BYTES_KEY } from "./bytes";
import { changedByRun, FS_GC_MS, FS_STALE_MS, makeQueryClient } from "./queryClient";

describe("the query client", () => {
  const qc = makeQueryClient();
  const defaultsFor = (key: unknown[]) => qc.getQueryDefaults(key);

  it("keeps a listing and a file answer rather than asking again on every mount", () => {
    // The tree re-expands, the pane reopens a file: on a remote store each of those is a
    // round trip, and nothing about the answer changed in between.
    expect(defaultsFor(["fs", "/notion/pages/x"]).staleTime).toBe(FS_STALE_MS);
    expect(defaultsFor(["file", "/notion/pages/x", "notion"]).staleTime).toBe(FS_STALE_MS);
    expect(defaultsFor(["fs", "/"]).gcTime).toBe(FS_GC_MS);
  });

  it("gives read bytes a life of their own", () => {
    // They are never stale on their own — `useBytes` says so — but they are the entries with
    // a size worth minding, so they are kept on a clock of their own.
    expect(defaultsFor([BYTES_KEY, "/a.pdf"]).gcTime).toBe(BYTES_GC_MS);
    expect(defaultsFor([BYTES_KEY, "/a.pdf"]).staleTime).toBeUndefined();
  });

  it("leaves everything else alone", () => {
    // A session list or a usage total is not a file answer; keeping one for half an hour
    // would show a conversation that has moved on.
    expect(defaultsFor(["sessions"]).staleTime).toBeUndefined();
    expect(defaultsFor(["messages", "abc"]).gcTime).toBeUndefined();
  });
});

describe("changedByRun", () => {
  it("takes the artifacts tree, which is where a run writes", () => {
    expect(changedByRun(["fs", "/artifacts"])).toBe(true);
    expect(changedByRun(["fs", "/artifacts/report"])).toBe(true);
    expect(changedByRun(["file", "/artifacts/report/out.md", "plain"])).toBe(true);
    expect(changedByRun(["bytes", "/artifacts/chart.png"])).toBe(true);
  });

  it("leaves the rest of the tree alone", () => {
    // The agent's console mounts the workspace as context, which refuses its writes. A
    // Notion row is a `file` query and a page render on the far side; re-reading every one
    // of them at the end of a run was the whole reason for this rule.
    expect(changedByRun(["file", "/notion/pages/A__1a2b/page.json", "notion"])).toBe(false);
    expect(changedByRun(["fs", "/notion/pages"])).toBe(false);
    expect(changedByRun(["fs", "/"])).toBe(false);
    expect(changedByRun(["bytes", "/s3/report.pdf"])).toBe(false);
  });

  it("is not about queries that are not files", () => {
    expect(changedByRun(["sessions"])).toBe(false);
    expect(changedByRun(["messages", "/artifacts"])).toBe(false);
    expect(changedByRun(["fs"])).toBe(false);
    // A path that only looks like the artifacts root is not under it.
    expect(changedByRun(["fs", "/artifacts-of-mine"])).toBe(false);
  });
});
