// The workspace view: whichever connected source the sidebar has open.
//
// The sources themselves — what is mounted, adding one, disconnecting one — are in the
// sidebar beside this, because they are navigation rather than content. That leaves the
// whole panel to the files.
//
// Every source is browsed through the same two commands. The engine mounts Notion, S3 and
// a local folder into one tree, so `fs_list` and `fs_read` already answer for all three
// and none of them needs a client of its own here. What differs is how the thing you open
// deserves to be shown, which is the one prop below.

import { useQuery } from "@tanstack/react-query";

import * as api from "@/api";
import { FileBrowser } from "@/components/FileBrowser";
import { SourceIcon } from "@/components/SourceIcon";
import { S } from "@/strings";
import type { MountInfo } from "@/types";

/** The root, for the first paint and for a selection whose source has gone away. */
const ROOT = "/";
/** The single directory cortex puts at the top of a Notion mount. */
const NOTION_PAGES = "pages";

export function WorkspacePanel({ source }: { source: string | null }) {
  // The same query the sidebar reads, so this is a cache hit rather than a second call.
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList });
  const rows: MountInfo[] = mounts.data ?? [];
  const path = source ?? ROOT;
  const open = rows.find((m) => m.path === path) ?? null;
  const notion = open?.kind === "notion";
  // A Notion mount's own root holds one directory, `pages`, and nothing else. Starting the
  // tree inside it puts the pages on screen at once instead of behind a folder that only
  // ever has one thing in it.
  const browseRoot = notion ? `${path}/${NOTION_PAGES}` : path;

  return (
    // `min-h-0 flex-1` rather than `h-full`: the banners above this in `main` are part of
    // the same column, and a full-height panel would push itself off the bottom by theirs.
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex items-center gap-2 px-6 pt-3 pb-4">
        {open && <SourceIcon kind={open.kind} className="size-5 shrink-0 text-muted-foreground" />}
        <h1 className="truncate text-xl font-semibold">{open?.label ?? S.workspace}</h1>
      </div>
      {/* Keyed on the root so switching sources drops the open file with the tree it came
          from: a selection under the old root names nothing under the new one. */}
      <FileBrowser key={browseRoot} root={browseRoot} kind={notion ? "notion" : "plain"} />
    </section>
  );
}
