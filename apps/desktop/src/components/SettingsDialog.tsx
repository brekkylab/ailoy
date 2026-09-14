// Settings: provider API keys, the defaults a new run inherits, and where the app keeps
// its logs.
//
// API keys are write-only from the webview's point of view. They are typed into
// `type="password"` fields, held in component state only while the row is being edited,
// handed to `settings_set` once, and then dropped; the engine never hands one back — a
// saved key comes home as `has_key` plus a `key_hint` suffix, and that is all this file
// ever renders.
//
// Every write goes through the one `save` mutation so that the engine's validation —
// which happens *before* anything is written, so a rejected patch changes nothing — has
// a single place to land (`api.messageOf`, at the bottom of the dialog).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";

import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { S } from "@/strings";
import type { ProviderSetting, SettingsPatch } from "@/types";

/** Only a placeholder: the engine picks its own default when `bedrock_region` is unset. */
const DEFAULT_BEDROCK_REGION = "us-east-1";

/** Send a patch; `onDone` runs only if the engine accepted it. */
type Save = (patch: SettingsPatch, onDone?: () => void) => void;

/**
 * Which provider row a patch belongs to, or `null` for a patch that is not a row's —
 * the default model, the turn limits, the catalog switch. One shared mutation serves the
 * whole dialog, so this is what keeps a save on one row from greying out all of them.
 * A lone Bedrock region is that row's too: it is sent from the field under its key.
 */
function providerOf(patch: SettingsPatch | undefined): string | null {
  if (!patch) return null;
  const [first] = Object.keys(patch.provider_keys ?? {});
  if (first != null) return first;
  return patch.bedrock_region != null ? "bedrock" : null;
}

/**
 * One provider's row: the key field, save, clear — and, for Bedrock alone, the region
 * that a Bedrock key is useless without.
 */
function ProviderRow({
  provider,
  save,
  pending,
}: {
  provider: ProviderSetting;
  save: Save;
  pending: boolean;
}) {
  const id = useId();
  const [key, setKey] = useState("");
  const [region, setRegion] = useState("");
  const isBedrock = provider.key === "bedrock";
  const typedKey = key.trim();
  const typedRegion = region.trim();

  const saveKey = () => {
    if (!typedKey) return;
    const patch: SettingsPatch = { provider_keys: { [provider.key]: typedKey } };
    // A key and its region are one edit, so they go in one patch: the engine validates
    // the whole thing or writes none of it.
    if (isBedrock && typedRegion) patch.bedrock_region = typedRegion;
    save(patch, () => {
      setKey("");
      setRegion("");
    });
  };

  // A region typed on its own still has to land somewhere, so it saves on blur — but not
  // while a key is sitting in the field above it, because that key's Save button is about
  // to send both together and clearing the field out from under it would lose the key.
  const saveRegion = () => {
    if (!typedRegion || typedKey || typedRegion === (provider.region ?? "")) return;
    save({ bedrock_region: typedRegion }, () => setRegion(""));
  };

  return (
    // `minmax(0,1fr)` and not `1fr`: an input's automatic minimum size would otherwise
    // push a long provider label plus its key hint past the dialog's width.
    <div className="grid grid-cols-[130px_minmax(0,1fr)_auto_auto] items-center gap-2">
      <Label htmlFor={id} className="min-w-0">
        <span className="truncate">{provider.label}</span>
      </Label>
      <Input
        id={id}
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder={provider.has_key ? `${S.apiKey} · ${provider.key_hint}` : S.apiKey}
        value={key}
        onChange={(e) => setKey(e.target.value)}
      />
      <Button size="sm" disabled={!typedKey || pending} onClick={saveKey}>
        {S.saveKey}
      </Button>
      <Button
        size="sm"
        variant="ghost"
        disabled={!provider.has_key || pending}
        // The draft goes with the stored key: leaving a half-typed one behind in a field
        // whose placeholder has just lost its hint reads as if it were still saved.
        onClick={() => save({ provider_keys: { [provider.key]: null } }, () => setKey(""))}
      >
        {S.clearKey}
      </Button>
      {isBedrock && (
        <Input
          className="col-span-4"
          aria-label={S.region}
          placeholder={`${S.region} (${provider.region ?? DEFAULT_BEDROCK_REGION})`}
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          onBlur={saveRegion}
        />
      )}
    </div>
  );
}

/**
 * A number that saves when the field loses focus. The engine happily accepts `0` for
 * either of these, which would end every run before it produced anything, so the floor
 * is 1 here; anything else — blank, a decimal, unchanged — snaps back to the stored
 * value rather than sending a patch.
 *
 * The draft starts from `value` and is re-seeded by remounting (the caller keys this on
 * the stored number), which is why there is no effect syncing prop into state.
 */
function NumberSetting({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: number;
  onCommit: (n: number) => void;
}) {
  const id = useId();
  const [text, setText] = useState(() => String(value));

  const commit = () => {
    const n = Number(text);
    if (Number.isInteger(n) && n >= 1 && n !== value) onCommit(n);
    else setText(String(value));
  };

  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type="number"
        min={1}
        step={1}
        inputMode="numeric"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
      />
    </div>
  );
}

export function SettingsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient();
  // Nothing here is worth a round trip while the dialog is shut.
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet, enabled: open });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList, enabled: open });
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo, enabled: open });
  const refreshId = useId();

  const save = useMutation({
    mutationFn: (patch: SettingsPatch) => api.settingsSet(patch),
    onSuccess: (next) => {
      // `settings_set` answers with the whole new `Settings`, so seed the cache with it
      // and the dialog repaints before the refetch lands. The `models` invalidate is not
      // housekeeping: `available` is derived from which keys exist, and it is what takes
      // the no-key banner off `App.tsx` the moment the first key is saved.
      qc.setQueryData(["settings"], next);
      void qc.invalidateQueries({ queryKey: ["settings"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
  const commit: Save = (patch, onDone) => save.mutate(patch, { onSuccess: () => onDone?.() });
  // `variables` is the patch currently in flight, which is enough to name the one row
  // that should be disabled — no second piece of state to keep in step with the mutation.
  const pendingProvider = save.isPending ? providerOf(save.variables) : null;
  // A mutation and not a bare call, so a failure to open the folder has somewhere to go.
  const openLogs = useMutation({ mutationFn: api.openLogs });

  const s = settings.data;

  return (
    // Base UI (not Radix): `onOpenChange` is `(open, eventDetails) => void`.
    <Dialog open={open} onOpenChange={(o) => onOpenChange(o)}>
      {/* `sm:` on the width, or `dialog.tsx`'s own `sm:max-w-sm` wins above that breakpoint. */}
      <DialogContent className="max-h-[85vh] overflow-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{S.settings}</DialogTitle>
        </DialogHeader>

        <section className="space-y-3">
          <h3 className="text-sm font-medium">{S.providers}</h3>
          {(s?.providers ?? []).map((p) => (
            <ProviderRow key={p.key} provider={p} save={commit} pending={pendingProvider === p.key} />
          ))}
        </section>

        {s && (
          <section className="grid grid-cols-2 gap-3">
            <div className="col-span-2 space-y-1">
              <Label>{S.defaultModel}</Label>
              {/* Unavailable models stay listed but disabled: seeing the model you wanted
                  greyed out is the hint that its provider is missing a key. */}
              <Select
                value={s.default_model || null}
                onValueChange={(m) => {
                  if (m) commit({ default_model: m });
                }}
              >
                <SelectTrigger className="w-full" aria-label={S.defaultModel}>
                  {/* Same reason as `Composer`: Base UI reads the trigger's text from the
                      selected *item*, and the items are in a portal that has not mounted
                      until the list is opened — so the stored default would show as a bare
                      id until then, while the open list showed `provider · name`. */}
                  <SelectValue placeholder={S.defaultModel}>
                    {(id: unknown) => {
                      if (typeof id !== "string" || !id) return S.defaultModel;
                      const m = models.data?.find((x) => x.id === id);
                      return m ? `${m.provider} · ${m.name}` : id;
                    }}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {(models.data ?? []).map((m) => (
                    <SelectItem key={m.id} value={m.id} disabled={!m.available}>
                      {m.provider} · {m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {/* Keyed on the stored value: when the engine answers with a different one
                the field remounts and re-seeds its draft from it. */}
            <NumberSetting
              key={`max_tokens:${s.max_tokens}`}
              label={S.maxTokens}
              value={s.max_tokens}
              onCommit={(n) => commit({ max_tokens: n })}
            />
            <NumberSetting
              key={`max_turns:${s.max_turns}`}
              label={S.maxTurns}
              value={s.max_turns}
              onCommit={(n) => commit({ max_turns: n })}
            />
            <div className="col-span-2 flex items-center gap-2">
              {/* Base UI: `onCheckedChange` is `(checked, eventDetails) => void`. */}
              <Switch
                id={refreshId}
                checked={s.catalog_refresh}
                onCheckedChange={(c) => commit({ catalog_refresh: c })}
              />
              <Label htmlFor={refreshId}>{S.catalogRefresh}</Label>
            </div>
          </section>
        )}

        <section className="space-y-1 border-t pt-3 text-xs text-muted-foreground">
          {ws.data && (
            <div>
              {S.workspace}: <span className="font-mono">{ws.data.mountpoint}</span> ·{" "}
              {ws.data.status.status === "mounted"
                ? S.mounted
                : `${S.degradedShort} (${ws.data.status.reason})`}
            </div>
          )}
          <Button
            size="sm"
            variant="link"
            className="h-auto px-0"
            onClick={() => openLogs.mutate()}
            disabled={openLogs.isPending}
          >
            {S.openLogs}
          </Button>
        </section>

        {save.isError && <p className="text-sm text-destructive">{api.messageOf(save.error)}</p>}
        {openLogs.isError && <p className="text-sm text-destructive">{api.messageOf(openLogs.error)}</p>}
      </DialogContent>
    </Dialog>
  );
}
