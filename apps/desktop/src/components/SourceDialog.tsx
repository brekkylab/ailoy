// What one connected source is, and the one thing you can do to it.
//
// For a connector that is disconnecting it. The engine has no command to change one in
// place — credentials are handed over once at `mount_add` and never read back, so "edit"
// would mean removing and re-adding, which is the two buttons the user already has.
//
// The root is the exception, and the only one: it is a local mount whose directory is a
// setting rather than a credential, so it can be repointed, and it cannot be removed —
// the workspace has to have a root, and the engine refuses to detach `/`.

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Trash2 } from "lucide-react";
import { useState } from "react";

import * as api from "@/api";
import { BYTES_KEY } from "@/lib/bytes";
import { SourceIcon } from "@/components/SourceIcon";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { WORKSPACE_ROOT } from "@/paths";
import { S } from "@/strings";
import type { MountInfo } from "@/types";

/** One `label: value` line of the sheet. */
function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[7rem_minmax(0,1fr)] items-baseline gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="truncate font-mono text-xs" title={value}>
        {value}
      </span>
    </div>
  );
}

/** The root's directory, with the picker that changes it. */
function RootDirectory({ source, onDone }: { source: MountInfo; onDone: () => void }) {
  const qc = useQueryClient();
  // Seeded from the source and not kept in sync with it: the box is what the user is
  // editing, and a poll landing mid-edit must not overwrite what they typed.
  const [dir, setDir] = useState(source.detail);
  const save = useMutation({
    mutationFn: (path: string) => api.workspaceSetRoot(path),
    onSuccess: () => {
      // Everything below `/` is a different tree now: the listings, any open preview, and
      // the row itself, which carries the directory as its detail.
      qc.invalidateQueries({ queryKey: ["mounts"] });
      qc.invalidateQueries({ queryKey: ["workspace"] });
      qc.invalidateQueries({ queryKey: ["fs"] });
      qc.invalidateQueries({ queryKey: ["file"] });
      qc.invalidateQueries({ queryKey: [BYTES_KEY] });
      onDone();
    },
  });

  const browse = async () => {
    const picked = await open({ directory: true, multiple: false });
    if (typeof picked === "string") setDir(picked);
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input value={dir} onChange={(e) => setDir(e.target.value)} aria-label={S.rootDirectory} />
        <Button variant="outline" size="icon" onClick={() => void browse()} aria-label={S.selectFolder}>
          <FolderOpen className="size-4" />
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">{S.rootDirectoryHint}</p>
      {save.isError && <p className="text-xs text-destructive">{api.messageOf(save.error)}</p>}
      <Button
        size="sm"
        disabled={save.isPending || dir.trim() === "" || dir === source.detail}
        onClick={() => save.mutate(dir.trim())}
      >
        {S.saveKey}
      </Button>
    </div>
  );
}

export function SourceDialog({ source, onClose }: { source: MountInfo | null; onClose: () => void }) {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: (path: string) => api.mountRemove(path),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mounts"] });
      qc.invalidateQueries({ queryKey: ["fs"] });
      // The preview may be showing a file inside the source that just went away.
      qc.invalidateQueries({ queryKey: ["file"] });
      qc.invalidateQueries({ queryKey: [BYTES_KEY] });
      onClose();
    },
  });
  const isRoot = source?.path === WORKSPACE_ROOT;

  return (
    // Base UI (not Radix): `onOpenChange` is `(open, eventDetails) => void`.
    <Dialog
      open={source !== null}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DialogContent>
        {source && (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <SourceIcon kind={source.kind} root={isRoot} className="size-4 shrink-0" />
                <span className="truncate">{source.label}</span>
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-2">
              <Row label={S.mountPath} value={source.path} />
              <Row label={S.sourceKind} value={source.kind} />
              {!isRoot && <Row label={S.sourceDetail} value={source.detail} />}
              {!source.writable && <Row label={S.readOnly} value="yes" />}
              {source.status.status === "error" && (
                <p className="text-xs text-destructive">
                  {S.errorPrefix}: {source.status.message}
                </p>
              )}
            </div>
            {isRoot && <RootDirectory source={source} onDone={onClose} />}
            <DialogFooter className="sm:justify-between">
              {isRoot ? (
                <span className="text-xs text-muted-foreground">{S.rootUndetachable}</span>
              ) : (
                <div className="flex min-w-0 flex-col gap-1">
                  <Button
                    variant="destructive"
                    size="sm"
                    className="self-start"
                    disabled={remove.isPending}
                    onClick={() => remove.mutate(source.path)}
                  >
                    <Trash2 className="size-3.5" /> {S.removeSource}
                  </Button>
                  <span className="text-xs text-muted-foreground">{S.removeSourceHint}</span>
                  {remove.isError && (
                    <span className="text-xs text-destructive">{api.messageOf(remove.error)}</span>
                  )}
                </div>
              )}
              <Button variant="outline" size="sm" onClick={onClose}>
                {S.close}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
