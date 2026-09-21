// What one connected source is, and the button that disconnects it.
//
// A read-only sheet rather than an edit form. The engine has no command to change a mount
// in place — a connector's credentials are handed over once at `mount_add` and never read
// back, so "edit" would mean removing and re-adding, which is the two buttons the user
// already has. What this shows is what `mount_list` returns: where it is mounted, what
// kind it is, and the redacted `detail` the engine is willing to say about it.

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";

import * as api from "@/api";
import { SourceIcon } from "@/components/SourceIcon";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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

export function SourceDialog({ source, onClose }: { source: MountInfo | null; onClose: () => void }) {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: (path: string) => api.mountRemove(path),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mounts"] });
      qc.invalidateQueries({ queryKey: ["fs"] });
      // The preview may be showing a file inside the source that just went away.
      qc.invalidateQueries({ queryKey: ["file"] });
      onClose();
    },
  });

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
                <SourceIcon kind={source.kind} className="size-4 shrink-0" />
                <span className="truncate">{source.label}</span>
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-2">
              <Row label={S.mountPath} value={source.path} />
              <Row label={S.sourceKind} value={source.kind} />
              <Row label={S.sourceDetail} value={source.detail} />
              {!source.writable && <Row label={S.readOnly} value="yes" />}
              {source.status.status === "error" && (
                <p className="text-xs text-destructive">
                  {S.errorPrefix}: {source.status.message}
                </p>
              )}
            </div>
            <DialogFooter className="sm:justify-between">
              {/* The root is the workspace's own files; there is nothing to disconnect it
                  from, and the engine refuses it anyway. */}
              {source.kind === "root" ? (
                <span className="text-xs text-muted-foreground">{S.workspace}</span>
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
