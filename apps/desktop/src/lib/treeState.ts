// Which directories are open, across a restart.
//
// The tree used to keep this per level, so collapsing a parent unmounted its children and
// forgot everything under them. That is fine for a tree that only has to last as long as
// the window; it cannot be saved, because by the time you would write it down most of it
// no longer exists. So expansion is one set of paths, owned above the tree.
//
// Per root, because a path means nothing under a different source — and because opening a
// bucket should not be able to restore a shape from someone's home directory.

/** How many open directories are remembered. */
const LIMIT = 500;

const key = (root: string) => `ailoy.tree:${root}`;

/**
 * The paths open under `root` when this was last used.
 *
 * Storage may be unavailable, hold something that is not ours, or hold a shape written by
 * an older version. All three read as "nothing was open", which is the one answer that is
 * never wrong — a tree that starts collapsed is a tree, while a tree built from a
 * half-understood blob is a crash on first paint.
 */
export function loadExpanded(root: string): Set<string> {
  try {
    const raw = localStorage.getItem(key(root));
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((p): p is string => typeof p === "string"));
  } catch {
    return new Set();
  }
}

/**
 * Remember what is open under `root`.
 *
 * Capped, and the cap keeps the *deepest* paths rather than the first ones: a tree is
 * opened downwards, so the long paths are the ones the user worked to reach, and dropping
 * a parent while keeping its child would restore neither.
 */
export function saveExpanded(root: string, expanded: Set<string>): void {
  try {
    let paths = [...expanded];
    if (paths.length > LIMIT) {
      paths = paths
        .sort((a, b) => a.split("/").length - b.split("/").length)
        .slice(-LIMIT);
    }
    localStorage.setItem(key(root), JSON.stringify(paths));
  } catch {
    /* storage may be unavailable */
  }
}

/**
 * `expanded` with `path` flipped.
 *
 * Closing a directory closes what was open inside it. Otherwise reopening it would spring
 * back to a shape the user collapsed on purpose, and the saved set would keep growing with
 * paths nothing can reach.
 */
export function toggle(expanded: Set<string>, path: string): Set<string> {
  const next = new Set(expanded);
  if (!next.delete(path)) {
    next.add(path);
    return next;
  }
  const inside = `${path}/`;
  for (const p of next) if (p.startsWith(inside)) next.delete(p);
  return next;
}

/** Whether a name is hidden by convention: a leading dot, as every unix tool reads it. */
export function isHidden(name: string): boolean {
  return name.startsWith(".");
}

/**
 * `expanded` with every directory between `root` and `path` opened.
 *
 * For a selection that did not come from the tree — a link inside a page, say. Following
 * one should leave the reader looking at where they landed, not at a tree still showing
 * where they were.
 */
export function expandTo(expanded: Set<string>, root: string, path: string): Set<string> {
  if (!path.startsWith(root)) return expanded;
  const next = new Set(expanded);
  const rest = path.slice(root.length).split("/").filter(Boolean);
  let at = root === "/" ? "" : root.replace(/\/$/, "");
  // The last segment is the target itself, which is opened by being selected, not by the
  // tree — a file has nothing to open, and a page opens whether or not it has children.
  for (const seg of rest.slice(0, -1)) {
    at = `${at}/${seg}`;
    next.add(at);
  }
  return next;
}
