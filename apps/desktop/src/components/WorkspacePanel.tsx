// The workspace view: whichever connected source the sidebar has open.
//
// The sources themselves — what is mounted, adding one, disconnecting one — are in the
// sidebar beside this, because they are navigation rather than content. That leaves the
// whole panel to the files.
//
// Every source is browsed through the same two commands. The engine mounts Notion, S3 and
// a local folder into one tree, so `fs_list` and `fs_read` already answer for all three
// and none of them needs a client of its own here. What differs is how the thing you open
// deserves to be shown, and — at the root — which of its children belong to it at all.

import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import * as api from "@/api";
import { FileBrowser } from "@/components/FileBrowser";
import { SourceIcon } from "@/components/icons/SourceIcon";
import { ARTIFACTS_ROOT, WORKSPACE_ROOT } from "@/paths";
import { S } from "@/strings";
import type { Entry, MountInfo } from "@/types";

/** The single directory cortex puts at the top of a Notion mount. */
const NOTION_PAGES = "pages";

export function WorkspacePanel({ source }: { source: string | null }) {
  // The same query the sidebar reads, so this is a cache hit rather than a second call.
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList });
  const rows: MountInfo[] = mounts.data ?? [];
  const path = source ?? WORKSPACE_ROOT;
  const open = rows.find((m) => m.path === path) ?? null;
  const notion = open?.kind === "notion";
  // A Notion mount's own root holds one directory, `pages`, and nothing else. Starting the
  // tree inside it puts the pages on screen at once instead of behind a folder that only
  // ever has one thing in it.
  const browseRoot = notion ? `${path}/${NOTION_PAGES}` : path;

  // Everything else is grafted into the root, so a listing of `/` returns the user's own
  // files *and* a directory per connector, plus the agent's artifacts. My Computer is the
  // machine's own files: the grafts have their own rows in the sidebar and their own place
  // in the nav, and showing them here again would say the user's disk contains Notion.
  //
  // Matched on the full path rather than the name, so a folder of the user's that happens
  // to be called `notion` deeper in the tree is still theirs and still shown.
  //
  // Keyed on `mounts.data` rather than on `rows`: the `?? []` above builds a new array on
  // every render while the query is empty, and a memo over that never holds — which would
  // hand `FileBrowser` a new predicate each time and remount the tree under it.
  const grafted = useMemo(() => {
    const paths = new Set<string>([ARTIFACTS_ROOT]);
    for (const m of mounts.data ?? []) if (m.path !== WORKSPACE_ROOT) paths.add(m.path);
    return paths;
  }, [mounts.data]);
  const hide = useMemo(
    () => (path === WORKSPACE_ROOT ? (e: Entry) => grafted.has(e.path) : undefined),
    [path, grafted],
  );

  return (
    // `min-h-0 flex-1` rather than `h-full`: the banners above this in `main` are part of
    // the same column, and a full-height panel would push itself off the bottom by theirs.
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex items-center gap-2 px-6 pt-3 pb-4">
        {open && (
          <SourceIcon
            kind={open.kind}
            root={open.path === WORKSPACE_ROOT}
            className="size-5 shrink-0 text-muted-foreground"
          />
        )}
        <h1 className="truncate text-xl font-semibold">{open?.label ?? S.workspace}</h1>
      </div>
      {/* Keyed on the root so switching sources drops the open file with the tree it came
          from: a selection under the old root names nothing under the new one. */}
      <FileBrowser
        key={browseRoot}
        root={browseRoot}
        kind={notion ? "notion" : "plain"}
        hide={hide}
        // Local folders and buckets only, for now. Notion renders its own pages, and what
        // sits under the workspace root is either one of these or something the agent
        // wrote, which the artifacts view shows as written.
        viewers={open?.kind === "local" || open?.kind === "s3"}
        // A leading dot only means "hidden" on a disk. A bucket key or a Notion title that
        // starts with one is just a name, and hiding it would lose the object.
        hiddenFiles={open?.kind === "local"}
      />
    </section>
  );
}
