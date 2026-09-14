// Connect dialogs: one form per mount kind (local folder, Notion, S3).
//
// Secrets — the Notion integration token, the S3 secret access key — are typed into
// `type="password"` inputs, held in state only for as long as the form is open, and
// handed to `mount_add` once. They are never logged, echoed back, or read again: the
// engine keeps them, and `mount_list` only ever returns a redacted `detail`.

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { type ReactNode, useId, useState } from "react";

import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { S } from "@/strings";
import type { MountConfig } from "@/types";

type Kind = "local" | "notion" | "s3";

/** A labelled input, with the label actually bound to it. `action` sits to its right. */
function Field({
  label,
  value,
  onChange,
  type,
  placeholder,
  readOnly,
  action,
}: {
  label: string;
  value: string;
  onChange?: (v: string) => void;
  type?: "text" | "password";
  placeholder?: string;
  readOnly?: boolean;
  action?: ReactNode;
}) {
  const id = useId();
  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      <div className="flex gap-2">
        <Input
          id={id}
          type={type ?? "text"}
          value={value}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={(e) => onChange?.(e.target.value)}
        />
        {action}
      </div>
    </div>
  );
}

/**
 * The dialog shell. Kept mounted so `open` can drive Base UI's open/close transition;
 * the form itself is mounted per opening and keyed by kind, so switching kinds — or
 * reopening the same one — never inherits the previous form's fields or its secret.
 */
export function MountDialog({ kind, onClose }: { kind: Kind | null; onClose: () => void }) {
  return (
    // Base UI (not Radix): `onOpenChange` is `(open, eventDetails) => void`.
    <Dialog
      open={kind !== null}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      {/* S3 asks for six fields in two columns; the default `sm:max-w-sm` wraps its labels. */}
      <DialogContent className={kind === "s3" ? "sm:max-w-md" : undefined}>
        {kind !== null && <MountForm key={kind} kind={kind} onClose={onClose} />}
      </DialogContent>
    </Dialog>
  );
}

function MountForm({ kind, onClose }: { kind: Kind; onClose: () => void }) {
  const qc = useQueryClient();
  const [path, setPath] = useState("");
  const [label, setLabel] = useState("");
  const [hostRoot, setHostRoot] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [s3, setS3] = useState({
    bucket: "",
    region: "us-east-1",
    access_key_id: "",
    secret_access_key: "",
    endpoint: "",
    key_prefix: "",
  });

  const add = useMutation({
    mutationFn: (config: MountConfig) => api.mountAdd({ path, label: label || null, config }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["mounts"] });
      qc.invalidateQueries({ queryKey: ["fs"] });
      qc.invalidateQueries({ queryKey: ["file"] });
      onClose();
    },
  });

  const pickFolder = async () => {
    const dir = await open({ directory: true, multiple: false });
    if (typeof dir === "string") {
      setHostRoot(dir);
      // A sensible default only: the last path segment, still editable above.
      if (!path) {
        const name = dir.split("/").filter(Boolean).pop();
        if (name) setPath("/" + name);
      }
    }
  };

  const submit = () => {
    if (kind === "local") add.mutate({ kind: "local", host_root: hostRoot });
    else if (kind === "notion") add.mutate({ kind: "notion", api_key: apiKey });
    else add.mutate({ ...s3, kind: "s3", endpoint: s3.endpoint || null, key_prefix: s3.key_prefix || null });
  };

  // The engine rejects an incomplete mount anyway; disabling here just says so sooner.
  const ready =
    path.trim() !== "" &&
    (kind === "local" ? hostRoot !== "" : kind === "notion" ? apiKey !== "" : s3.bucket !== "" && s3.region !== "");

  const title = kind === "local" ? S.connectLocal : kind === "notion" ? S.connectNotion : S.connectS3;
  return (
    <>
      <DialogHeader>
        <DialogTitle>{title}</DialogTitle>
      </DialogHeader>
      <div className="space-y-3">
        <Field label={S.path} value={path} onChange={setPath} placeholder="/docs" />
        <Field label={S.label} value={label} onChange={setLabel} />
        {kind === "local" && (
          <Field
            label={S.folder}
            value={hostRoot}
            readOnly
            placeholder="~/Documents/project"
            action={
              <Button variant="outline" size="icon" aria-label={S.selectFolder} onClick={() => void pickFolder()}>
                …
              </Button>
            }
          />
        )}
        {kind === "notion" && <Field label={S.notionToken} type="password" value={apiKey} onChange={setApiKey} />}
        {kind === "s3" && (
          <div className="grid grid-cols-2 gap-2">
            <Field label={S.bucket} value={s3.bucket} onChange={(v) => setS3({ ...s3, bucket: v })} />
            <Field label={S.region} value={s3.region} onChange={(v) => setS3({ ...s3, region: v })} />
            <Field label={S.accessKeyId} value={s3.access_key_id} onChange={(v) => setS3({ ...s3, access_key_id: v })} />
            <Field
              label={S.secretAccessKey}
              type="password"
              value={s3.secret_access_key}
              onChange={(v) => setS3({ ...s3, secret_access_key: v })}
            />
            <Field label={S.endpoint} value={s3.endpoint} onChange={(v) => setS3({ ...s3, endpoint: v })} />
            <Field label={S.keyPrefix} value={s3.key_prefix} onChange={(v) => setS3({ ...s3, key_prefix: v })} />
          </div>
        )}
        {add.isError && <p className="text-sm text-destructive">{api.messageOf(add.error)}</p>}
      </div>
      <DialogFooter>
        <Button variant="ghost" onClick={onClose}>
          {S.cancel}
        </Button>
        <Button onClick={submit} disabled={add.isPending || !ready}>
          {S.connect}
        </Button>
      </DialogFooter>
    </>
  );
}
