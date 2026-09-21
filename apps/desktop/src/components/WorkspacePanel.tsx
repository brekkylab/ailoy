// The workspace view: what is mounted, and the tree over it.
//
// This fills the main panel, reached from the sidebar, rather than sitting in a column of
// its own, and names itself: the title bar is left to conversations, so there is room here
// to set the heading at a size that reads as the page's rather than the window's.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Cloud, FolderPlus, Globe, HardDrive, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import * as api from "@/api";
import { FileBrowser } from "@/components/FileBrowser";
import { MountDialog } from "@/components/MountDialogs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { S } from "@/strings";

export function WorkspacePanel() {
  const qc = useQueryClient();
  // Polled: connectors are restored in the background after launch, so `mount_list` is
  // eventually consistent and there is no event to invalidate on. Three seconds is short
  // enough that a restored mount appears while the user is still looking at the panel.
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList, refetchInterval: 3000 });
  const [dialog, setDialog] = useState<"local" | "notion" | "s3" | null>(null);
  const remove = useMutation({
    mutationFn: (p: string) => api.mountRemove(p),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mounts"] });
      qc.invalidateQueries({ queryKey: ["fs"] });
      // The preview may be showing a file inside the mount that just went away.
      qc.invalidateQueries({ queryKey: ["file"] });
    },
  });

  // Each mount is a directory at the root, so the root listing is stale the moment the
  // polled list changes. Keyed on a signature rather than on `mounts.data` itself: the
  // poll hands back a new array every three seconds, and refetching the root that often
  // would keep a connector listing over the network.
  // Printable delimiters on purpose: paths cannot contain a newline after normalization,
  // and control bytes in source turn the file binary for git.
  const signature = (mounts.data ?? []).map((m) => `${m.path}=${m.status.status}`).join("\n");
  useEffect(() => {
    qc.invalidateQueries({ queryKey: ["fs", "/"] });
  }, [signature, qc]);

  return (
    // `min-h-0 flex-1` rather than `h-full`: the banners above this in `main` are part of
    // the same column, and a full-height panel would push itself off the bottom by theirs.
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex items-center justify-between px-6 pt-3 pb-4">
        <h1 className="text-xl font-semibold">{S.workspace}</h1>
        <DropdownMenu>
          {/* Base UI (not Radix): the trigger takes `render`, not `asChild`. */}
          <DropdownMenuTrigger render={<Button size="sm" variant="outline" />}>
            <FolderPlus className="size-3.5" />
            {S.addMount}
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-auto">
            <DropdownMenuItem onClick={() => setDialog("local")}>
              <HardDrive /> {S.connectLocal}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setDialog("notion")}>
              <Globe /> {S.connectNotion}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setDialog("s3")}>
              <Cloud /> {S.connectS3}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="space-y-1 px-6 pb-3">
        {(mounts.data ?? []).map((m) => (
          <div key={m.path} className="flex items-center gap-2 text-xs">
            <Badge
              variant={m.status.status === "error" ? "destructive" : "secondary"}
              title={m.status.status === "error" ? m.status.message : undefined}
            >
              {m.kind}
            </Badge>
            <span className="truncate font-mono" title={m.detail}>
              {m.path}
            </span>
            {!m.writable && <span className="shrink-0 text-muted-foreground">{S.readOnly}</span>}
            {m.status.status === "error" && (
              <span className="shrink-0 text-destructive" title={m.status.message} aria-label={S.errorPrefix}>
                !
              </span>
            )}
            {/* The root is the workspace itself; only connectors can be disconnected. */}
            {m.kind !== "root" && (
              <button
                className="ml-auto shrink-0 text-muted-foreground hover:text-foreground"
                aria-label={S.remove}
                disabled={remove.isPending && remove.variables === m.path}
                onClick={() => remove.mutate(m.path)}
              >
                <Trash2 className="size-3.5" />
              </button>
            )}
          </div>
        ))}
        {mounts.isError && <p className="text-xs text-destructive">{api.messageOf(mounts.error)}</p>}
        {remove.isError && <p className="text-xs text-destructive">{api.messageOf(remove.error)}</p>}
      </div>
      <FileBrowser root="/" />
      <MountDialog kind={dialog} onClose={() => setDialog(null)} />
    </section>
  );
}
