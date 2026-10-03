import { describe, expect, it } from "vitest";

import type { MountInfo } from "@/types";

import { placeOf, placeWorkspacePath } from "./workspacePath";

const INFO = { mountpoint: "/Users/me/Library/Application Support/ailoy/workspace", files_root: "/Users/me" };
const mount = (path: string, kind: MountInfo["kind"] = "local"): MountInfo => ({
  id: path,
  path,
  kind,
  label: path,
  detail: "",
  writable: false,
  status: { status: "ok" },
});
const MOUNTS = [mount("/"), mount("/notion", "notion"), mount("/notion-archive")];

describe("placeOf", () => {
  it("places a path through the mount under My Computer, or under the source that holds it", () => {
    expect(placeOf(`${INFO.mountpoint}/docs/a.md`, INFO, MOUNTS)).toEqual({ tab: "files", source: null, path: "/docs/a.md" });
    expect(placeOf(`${INFO.mountpoint}/notion/pages/x/page.json`, INFO, MOUNTS)).toEqual({
      tab: "files",
      source: "/notion",
      path: "/notion/pages/x/page.json",
    });
    // A source whose name only starts like another's is its own.
    expect(placeOf(`${INFO.mountpoint}/notion-archive/b`, INFO, MOUNTS)?.source).toBe("/notion-archive");
  });

  it("places the agent's own output under Artifacts, by either spelling", () => {
    const art = "/Users/me/Library/Application Support/ailoy/artifacts/report.md";
    expect(placeOf(art, INFO, MOUNTS)).toEqual({ tab: "artifacts", source: null, path: "/artifacts/report.md" });
    expect(placeOf(`${INFO.mountpoint}/artifacts/report.md`, INFO, MOUNTS)?.tab).toBe("artifacts");
  });

  it("reads the root directly when the workspace is not mounted", () => {
    expect(placeOf("/Users/me/notes/todo.md", INFO, MOUNTS)).toEqual({ tab: "files", source: null, path: "/notes/todo.md" });
  });

  it("has nowhere to put what the window cannot open", () => {
    expect(placeOf("/tmp/x", INFO, MOUNTS)).toBeNull();
    expect(placeOf("relative/path", INFO, MOUNTS)).toBeNull();
    expect(placeOf("/Users/me/a", undefined, MOUNTS)).toBeNull();
  });

  it("places a workspace path the same way", () => {
    expect(placeWorkspacePath("/artifacts", MOUNTS).tab).toBe("artifacts");
    expect(placeWorkspacePath("/notion", MOUNTS).source).toBe("/notion");
  });
});
