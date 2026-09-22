// Reading one node of a Notion tree.
//
// A page directory holds `page.json` and a database directory holds `database.json`, and
// the directory's own name says which — cortex puts `__db__` in one and not the other so a
// path there resolves without fetching its parent. Asking for the right file outright is
// what that marker is for; trying one and falling back to the other would spend a refused
// request on every database in the tree.
//
// One query key, `["file", <dir>, "notion"]`, shared by everything that reads a node: the
// tree row that wants its icon, the panel that renders its body, and the links in a parent
// page. A row the user then clicks is already in hand, and nothing is fetched twice. The
// tree and the viewer both read a node, so the query lives here rather than beside either.

import type { UseQueryOptions } from "@tanstack/react-query";

import * as api from "@/api";
import { isNotionDatabase } from "@/lib/notion";
import type { FileContent } from "@/types";

/** The json behind a page or database directory. */
export function readNotionNode(path: string): Promise<FileContent> {
  const name = path.slice(path.lastIndexOf("/") + 1);
  return api.fsRead(`${path}/${isNotionDatabase(name) ? "database.json" : "page.json"}`);
}

/**
 * The shared query for a node.
 *
 * `staleTime: Infinity` because a render costs cortex a page fetch and a walk of its whole
 * block tree — worth paying when a row first appears, not again every time the window is
 * focused. A failure is not retried for the same reason: the row falls back to the plain
 * page icon, which is a better answer than three more requests.
 */
export function notionNodeQuery(path: string) {
  return {
    queryKey: ["file", path, "notion"],
    queryFn: () => readNotionNode(path),
    staleTime: Infinity,
    retry: false,
  } satisfies UseQueryOptions<FileContent>;
}
