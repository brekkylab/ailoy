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
// page. A row the user then clicks is already in hand, and nothing is fetched twice.

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import * as api from "@/api";
import type { RowInfo, TreeAdapter } from "@/components/FileTree";
import {
  DB_ICON,
  PAGE_ICON,
  isNotionDatabase,
  notionChildren,
  notionHeading,
  notionIcon,
  notionRowCount,
  notionTitle,
} from "@/lib/notion";
import { recalledRow, rememberRow } from "@/lib/notionRows";
import type { Entry, FileContent } from "@/types";

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

/**
 * A row's icon and whether it holds anything.
 *
 * Both come from the node's own json, so the tree reads one per visible row. That is the
 * price of a page tree that looks like Notion's: an icon is not in the parent's listing,
 * nor in the `child_page` block the parent was built from — Notion puts it on the page
 * object and nowhere else, so whoever wants it has to open the page. The queries are the
 * ones clicking those rows would make anyway, and they are kept rather than refetched.
 */
function useNotionRow(e: Entry): RowInfo {
  const node = useQuery({ ...notionNodeQuery(e.path), enabled: e.kind === "dir" });
  const db = isNotionDatabase(e.name);
  // A page.json past the read cap comes back cut, and cut json does not parse. Its icon is
  // simply missing; its children must read as unknown rather than as none, or a long page
  // would lose the sub-pages it does have.
  const text = node.data && !node.data.truncated ? node.data.text : null;
  // What this row was last time, which is what it is drawn as until the read lands. A
  // restart empties the query cache but not this, so the tree comes back with its icons and
  // its chevrons rather than filling them in one row at a time. See `lib/notionRows`.
  const recalled = useMemo(() => (e.kind === "dir" ? recalledRow(e.path) : null), [e.kind, e.path]);
  const icon = text ? notionIcon(text) : null;
  // The page's own title, which is the one with the spaces in it: cortex names the directory
  // `<sanitized-title>__<id>`, because that is a path component and a path component cannot
  // hold every character a Notion title can. The json inside carries the real one, and this
  // row is already reading it.
  const title = text ? notionHeading(text) : null;
  const leaf = text === null ? undefined : db ? notionRowCount(text) === 0 : notionChildren(text).length === 0;
  // Written after the render, not during it: this is a side effect, and one that reads the
  // same storage the row above it may be writing.
  useEffect(() => {
    if (leaf !== undefined) rememberRow(e.path, { icon, leaf, title });
  }, [e.path, icon, leaf, title]);
  return {
    icon: (
      <span className="w-3.5 shrink-0 text-center text-[13px] leading-none">
        {icon ?? recalled?.icon ?? (db ? DB_ICON : PAGE_ICON)}
      </span>
    ),
    label: title ?? recalled?.title ?? undefined,
    leaf: leaf ?? recalled?.leaf,
    // Until there is an answer — from the read, or from the last time this row was seen —
    // there is no expander: most pages hold nothing, so assuming otherwise showed a chevron
    // on nearly every row and then took it away. A read that *failed* is not pending: the
    // row keeps the expander, because a page whose json could not be read may still have
    // sub-pages behind it.
    pending: e.kind === "dir" && node.isPending && !recalled,
  };
}

/**
 * Notion's shape, in one place.
 *
 * A page is a directory holding `page.json`, so the directory *is* the page: it carries
 * the title, and clicking it is asking for the body. The json files are hidden because
 * they are that body rather than siblings of it, and the id cortex sanitizes into the
 * directory name comes off, since the tree is a list of pages and not of paths.
 */
export const NOTION: TreeAdapter = {
  label: (e) => (e.kind === "dir" ? notionTitle(e.name) : e.name),
  hide: (e) => e.name === "page.json" || e.name === "database.json",
  openDirs: true,
  useRow: useNotionRow,
};
