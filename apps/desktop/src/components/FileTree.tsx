// The workspace tree, one `fs_list` per expanded directory.
//
// Lazily loaded on purpose: a mounted connector (Notion, S3) lists over the network, so
// listing the whole tree up front would stall on directories nobody opened. Each level is
// its own component with its own `["fs", path]` query, which also gives the panel a
// precise key to invalidate when a connector appears at the root or goes away.

import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { ChevronRight, File, Folder } from "lucide-react";
import type { ReactNode } from "react";

import * as api from "@/api";
import { expandable } from "@/lib/treeState";
import { S } from "@/strings";
import type { Entry } from "@/types";

/** What a source wants on a row beyond its name. */
export type RowInfo = {
  /** Drawn where the folder or file icon would be. */
  icon?: ReactNode;
  /**
   * True when a directory has nothing inside, so no expander is offered for it. Absent is
   * "this source has no opinion", which leaves a directory expandable — it is one.
   */
  leaf?: boolean;
  /**
   * The source is still finding out, so no expander *yet*.
   *
   * Not the same as having no opinion, and the difference is what a page tree looks like:
   * a page is a directory because it may have sub-pages and usually has none, so assuming
   * expandable while the answer was in flight put a chevron on nearly every row and then
   * took it off again as each page's json arrived. The space is reserved either way, so a
   * chevron appearing costs no movement. See `lib/treeState`'s `expandable`.
   */
  pending?: boolean;
};

/** A source with nothing to add. Calls no hooks, which is what keeps `useRow` honest. */
const plainRow = (): RowInfo => ({});

/**
 * How a source wants its entries shown. Absent, the tree lists exactly what the engine
 * returns: files open, directories only expand. A source whose layout means something
 * more than files — Notion, where a page is a directory with the id sanitized into its
 * name — supplies one of these rather than teaching this component about it.
 */
export type TreeAdapter = {
  /** What to call an entry, when its name on disk is not what to read. */
  label?: (e: Entry) => string;
  /** Entries to leave out of the listing entirely. */
  hide?: (e: Entry) => boolean;
  /** True when clicking a directory should open it as well as expand it. */
  openDirs?: boolean;
  /**
   * Per-row decoration — a hook, because for Notion the answer is a fetch: a page's icon
   * and whether it holds any sub-pages both live in the `page.json` behind it.
   *
   * A hook in a plain object is a contract about shape: an adapter may be rebuilt as often
   * as it likes, but it must not gain or lose this field while a tree is mounted, or the
   * rows under it would change how many hooks they call. Nothing does — `kind` is fixed
   * per source, and `WorkspacePanel` keys the whole browser on its root.
   */
  useRow?: (e: Entry) => RowInfo;
};

export function FileTree({
  path,
  depth = 0,
  onOpen,
  selected,
  adapter,
  expanded,
  onToggle,
}: {
  path: string;
  depth?: number;
  onOpen: (path: string) => void;
  selected: string | null;
  adapter?: TreeAdapter;
  /**
   * Every open directory, for the whole tree, owned above it.
   *
   * Not per level: collapsing a parent unmounts its children, so state held there is gone
   * before anything could write it down. One set, held by the browser, is what lets the
   * shape outlive the window — see `lib/treeState`.
   */
  expanded: Set<string>;
  onToggle: (path: string) => void;
}) {
  const entries = useQuery({ queryKey: ["fs", path], queryFn: () => api.fsList(path) });

  const rows = (entries.data ?? []).filter((e) => !adapter?.hide?.(e));

  return (
    <ul>
      {rows.map((e) => (
        <Row
          key={e.path}
          entry={e}
          depth={depth}
          onOpen={onOpen}
          selected={selected}
          adapter={adapter}
          expanded={expanded}
          onToggle={onToggle}
        />
      ))}
      {rows.length === 0 && entries.data && (
        <li className="py-0.5 text-xs text-muted-foreground" style={{ paddingLeft: 4 + depth * 12 }}>
          {S.empty}
        </li>
      )}
      {entries.isError && (
        <li className="px-2 text-xs text-destructive" style={{ paddingLeft: 4 + depth * 12 }}>
          {api.messageOf(entries.error)}
        </li>
      )}
    </ul>
  );
}

/**
 * One entry, as its own component so a source's `useRow` has somewhere to run.
 *
 * It cannot be a call inside the loop above: a hook per row means a component per row.
 */
function Row({
  entry: e,
  depth,
  onOpen,
  selected,
  adapter,
  expanded,
  onToggle,
}: {
  entry: Entry;
  depth: number;
  onOpen: (path: string) => void;
  selected: string | null;
  adapter?: TreeAdapter;
  expanded: Set<string>;
  onToggle: (path: string) => void;
}) {
  const { icon, leaf, pending } = (adapter?.useRow ?? plainRow)(e);
  // A directory known to hold nothing is drawn as what it is, and one whose source has not
  // answered yet waits. For Notion that is most of them: a page is a directory because it
  // *may* have sub-pages, and usually has none.
  const open = expanded.has(e.path);
  const isExpandable = expandable({ isDir: e.kind === "dir", leaf, pending, open });

  return (
    <li>
      <button
        className={cn(
          "flex w-full items-center gap-1 truncate rounded px-1 py-0.5 text-left text-xs hover:bg-accent",
          selected === e.path && "bg-accent",
        )}
        style={{ paddingLeft: 4 + depth * 12 }}
        onClick={() => {
          // A directory that also opens does both on the one click: in a page tree the
          // children and the body are the same thing being asked for.
          if (isExpandable) onToggle(e.path);
          if (e.kind !== "dir" || adapter?.openDirs) onOpen(e.path);
        }}
        title={e.size != null ? `${e.size} B` : undefined}
      >
        {isExpandable ? (
          <ChevronRight className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")} />
        ) : (
          <span className="w-3 shrink-0" />
        )}
        {icon ??
          (e.kind === "dir" ? <Folder className="size-3.5 shrink-0" /> : <File className="size-3.5 shrink-0" />)}
        <span className="truncate">{adapter?.label?.(e) ?? e.name}</span>
      </button>
      {isExpandable && open && (
        <FileTree
          path={e.path}
          depth={depth + 1}
          onOpen={onOpen}
          selected={selected}
          adapter={adapter}
          expanded={expanded}
          onToggle={onToggle}
        />
      )}
    </li>
  );
}
