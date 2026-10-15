// Where a path the agent used is, in the window's own terms.
//
// The agent names everything by its absolute path on this machine — the workspace's mount
// point, the artifacts root — and the window names it by its path in the workspace and
// the place that shows it: a source under Files, or Artifacts. This is the one mapping from
// the first to the second, for anything that wants to open what a tool call touched.

import { ARTIFACTS_ROOT, WORKSPACE_ROOT } from "@/paths";
import type { MountInfo } from "@/types";

/** Where a file is shown: which tab, which source under Files, and its workspace path. */
export interface Place {
  tab: "files" | "artifacts";
  /** The source's mount path under Files — `/notion` — or `null` for My Computer. */
  source: string | null;
  path: string;
}

/** `rest` of `abs` under `root`, as a workspace path (`/a/b`), or `null` when it is not under it. */
function under(abs: string, root: string): string | null {
  const base = root.replace(/\/+$/, "");
  if (!base) return null;
  if (abs === base) return WORKSPACE_ROOT;
  if (!abs.startsWith(`${base}/`)) return null;
  return abs.slice(base.length);
}

/** A workspace path, placed: the artifacts tree, a connected source, or My Computer. */
export function placeWorkspacePath(path: string, mounts: MountInfo[]): Place {
  if (path === ARTIFACTS_ROOT || path.startsWith(`${ARTIFACTS_ROOT}/`)) {
    return { tab: "artifacts", source: null, path };
  }
  // The longest mount path that holds it, so `/notion` wins over the root for its pages.
  const owner = mounts
    .filter((m) => m.path !== WORKSPACE_ROOT && (path === m.path || path.startsWith(`${m.path}/`)))
    .sort((a, b) => b.path.length - a.path.length)[0];
  return { tab: "files", source: owner?.path ?? null, path };
}

/**
 * Where an absolute path the agent used is shown, or `null` for one outside everything the
 * window can open — a scratch file, `/tmp`, a path on the host the workspace does not hold.
 *
 * The artifacts root is tried first: it sits beside the mount point, and the workspace also
 * grafts it in at `/artifacts`, so either spelling lands on the same tab.
 */
export function placeOf(
  abs: string,
  info: { mountpoint: string; files_root: string } | undefined,
  mounts: MountInfo[] | undefined,
): Place | null {
  if (!info || !abs.startsWith("/")) return null;
  const dataDir = info.mountpoint.replace(/\/[^/]+\/?$/, "");
  const art = under(abs, `${dataDir}/artifacts`);
  if (art !== null) return { tab: "artifacts", source: null, path: art === WORKSPACE_ROOT ? ARTIFACTS_ROOT : `${ARTIFACTS_ROOT}${art}` };
  // Through the mount when the workspace is mounted, straight off the root when it is not.
  const ws = under(abs, info.mountpoint) ?? under(abs, info.files_root);
  return ws === null ? null : placeWorkspacePath(ws, mounts ?? []);
}
