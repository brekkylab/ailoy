// The workspace tree, one `fs_list` per expanded directory.
//
// Lazily loaded on purpose: a mounted connector (Notion, S3) lists over the network, so
// listing the whole tree up front would stall on directories nobody opened. Each level is
// its own component with its own `["fs", path]` query, which also gives the panel a
// precise key to invalidate when a connector appears at the root or goes away.

import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { ChevronRight, File, Folder } from "lucide-react";

import * as api from "@/api";
import { S } from "@/strings";
import type { Entry } from "@/types";

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
        <li key={e.path}>
          <button
            className={cn(
              "flex w-full items-center gap-1 truncate rounded px-1 py-0.5 text-left text-xs hover:bg-accent",
              selected === e.path && "bg-accent",
            )}
            style={{ paddingLeft: 4 + depth * 12 }}
            onClick={() => {
              // A directory that also opens does both on the one click: in a page tree the
              // children and the body are the same thing being asked for.
              if (e.kind === "dir") onToggle(e.path);
              if (e.kind !== "dir" || adapter?.openDirs) onOpen(e.path);
            }}
            title={e.size != null ? `${e.size} B` : undefined}
          >
            {e.kind === "dir" ? (
              <ChevronRight className={cn("size-3 shrink-0 transition-transform", expanded.has(e.path) && "rotate-90")} />
            ) : (
              <span className="w-3 shrink-0" />
            )}
            {e.kind === "dir" ? <Folder className="size-3.5 shrink-0" /> : <File className="size-3.5 shrink-0" />}
            <span className="truncate">{adapter?.label?.(e) ?? e.name}</span>
          </button>
          {e.kind === "dir" && expanded.has(e.path) && (
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
