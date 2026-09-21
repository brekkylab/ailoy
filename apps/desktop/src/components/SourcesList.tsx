// The connected sources, in the sidebar, while the workspace is the open panel.
//
// They live here rather than above the file tree because they are navigation furniture:
// what is plugged into the workspace, alongside what else the window can show. The tree
// then gets the whole main panel.

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "cn";
import { Cloud, FolderPlus, Globe, HardDrive, MoreHorizontal } from "lucide-react";
import { useEffect, useState } from "react";

import * as api from "@/api";
import { MountDialog } from "@/components/MountDialogs";
import { SourceDialog } from "@/components/SourceDialog";
import { SourceIcon } from "@/components/SourceIcon";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { WORKSPACE_ROOT } from "@/paths";
import { S } from "@/strings";
import type { MountInfo } from "@/types";

export function SourcesList({
  selected,
  onSelect,
}: {
  /** The source path the main panel is showing, or `null` for the root. */
  selected: string | null;
  onSelect: (path: string) => void;
}) {
  const qc = useQueryClient();
  // Polled: connectors are restored in the background after launch, so `mount_list` is
  // eventually consistent and there is no event to invalidate on. Three seconds is short
  // enough that a restored source appears while the user is still looking at the list.
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList, refetchInterval: 3000 });
  const [adding, setAdding] = useState<"local" | "notion" | "s3" | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  // Each source is a directory at the root, so the root listing is stale the moment the
  // polled list changes. Keyed on a signature rather than on `mounts.data` itself: the
  // poll hands back a new array every three seconds, and refetching the root that often
  // would keep a connector listing over the network.
  // Printable delimiters on purpose: paths cannot contain a newline after normalization,
  // and control bytes in source turn the file binary for git.
  const rows: MountInfo[] = mounts.data ?? [];
  const signature = rows.map((m) => `${m.path}=${m.status.status}`).join("\n");
  useEffect(() => {
    qc.invalidateQueries({ queryKey: ["fs", "/"] });
  }, [signature, qc]);

  // The open sheet is a path, and its contents are looked up in the live list rather than
  // copied into state — so a source that changes status, or goes away entirely while its
  // sheet is open, updates or closes instead of showing a snapshot of something that is no
  // longer there. Nothing has to clear `open` when that happens: with no row to find, the
  // sheet has nothing to show and closes itself, and the next click overwrites it.
  const sheet = rows.find((m) => m.path === open) ?? null;

  return (
    <>
      <div className="flex items-center justify-between px-3 pt-1 pb-2">
        <h2 className="text-xs font-medium text-muted-foreground">{S.mounts}</h2>
        <DropdownMenu>
          {/* Base UI (not Radix): the trigger takes `render`, not `asChild`. */}
          <DropdownMenuTrigger
            render={<Button size="icon" variant="ghost" className="size-6" aria-label={S.addMount} />}
          >
            <FolderPlus className="size-3.5" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-auto">
            <DropdownMenuItem onClick={() => setAdding("local")}>
              <HardDrive /> {S.connectLocal}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setAdding("notion")}>
              <Globe /> {S.connectNotion}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setAdding("s3")}>
              <Cloud /> {S.connectS3}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <ScrollArea className="min-h-0 flex-1 px-2">
        {rows.map((m) => (
          <div
            key={m.path}
            className={cn(
              "group flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
              // The root stands in for a null selection, so it lights up on the first paint
              // as the panel beside it already shows it.
              (selected ?? WORKSPACE_ROOT) === m.path && "bg-accent",
            )}
          >
            <SourceIcon
              kind={m.kind}
              root={m.path === WORKSPACE_ROOT}
              className="size-4 shrink-0 text-muted-foreground"
            />
            <button
              className="min-w-0 flex-1 truncate text-left"
              title={m.path}
              onClick={() => onSelect(m.path)}
            >
              {m.label}
            </button>
            {m.status.status === "error" && (
              <span className="shrink-0 text-destructive" title={m.status.message} aria-label={S.errorPrefix}>
                !
              </span>
            )}
            <button
              className={cn(
                "shrink-0 rounded p-1 text-muted-foreground opacity-0 transition-opacity",
                "hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100",
              )}
              aria-label={S.sourceOptions}
              onClick={() => setOpen(m.path)}
            >
              <MoreHorizontal className="size-3.5" />
            </button>
          </div>
        ))}
        {rows.length === 0 && !mounts.isLoading && (
          <p className="p-3 text-xs text-muted-foreground">{S.noSources}</p>
        )}
        {mounts.isError && <p className="p-3 text-xs text-destructive">{api.messageOf(mounts.error)}</p>}
      </ScrollArea>

      <MountDialog kind={adding} onClose={() => setAdding(null)} />
      <SourceDialog source={sheet} onClose={() => setOpen(null)} />
    </>
  );
}
