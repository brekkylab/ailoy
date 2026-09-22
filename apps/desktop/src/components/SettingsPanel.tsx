// Settings: provider API keys, the defaults a new run inherits, and where the app keeps
// its logs.
//
// A panel and not a dialog. Keys are the one thing a new install cannot start without, and
// a provider's pane is where the user finds out whether the key they pasted took — which is
// reading, comparing and coming back, not the one decision a modal is for. A modal also
// takes the window hostage: the no-key banner that sent the user here is behind it, and so
// is the model picker the key is about to fill.
//
// API keys are write-only from the webview's point of view. They are typed into
// `type="password"` fields, held in component state only while the row is being edited,
// handed to `settings_set` once, and then dropped; the engine never hands one back — a
// saved key comes home as `has_key` plus a `key_hint` suffix, and that is all this file
// ever renders.
//
// Every write goes through the one `save` mutation so that the engine's validation —
// which happens *before* anything is written, so a rejected patch changes nothing — has a
// single place to land (`api.messageOf`, under the section that sent it).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";

import * as api from "@/api";
import { ProviderIcon } from "@/components/ProviderIcon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { modelsOf, type ProviderModels } from "@/lib/providers";
import { S } from "@/strings";
import type { ProviderSetting, SettingsPatch } from "@/types";

/** Only a placeholder: the engine picks its own default when `bedrock_region` is unset. */
const DEFAULT_BEDROCK_REGION = "us-east-1";

/** How many model names a provider's pane lists before it counts the rest instead. */
const NAMED_MODELS = 8;

/** Send a patch; `onDone` runs only if the engine accepted it. */
type Save = (patch: SettingsPatch, onDone?: () => void) => void;

/**
 * Which provider pane a patch belongs to, or `null` for a patch that is not a pane's —
 * the default model, the turn limits, the catalog switch. One shared mutation serves the
 * whole panel, so this is what keeps a save on one provider from greying out the rest.
 * A lone Bedrock region is that pane's too: it is sent from the field under its key.
 */
function providerOf(patch: SettingsPatch | undefined): string | null {
  if (!patch) return null;
  const [first] = Object.keys(patch.provider_keys ?? {});
  if (first != null) return first;
  return patch.bedrock_region != null ? "bedrock" : null;
}

/** A section of the panel: a heading, and the rows under it. */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">{title}</h2>
      {children}
    </section>
  );
}

/**
 * One provider's settings: the key it is reached with, what that key unlocks, and — for
 * Bedrock alone — the region a Bedrock key is useless without.
 *
 * Unmounted with its tab, which is what resets the drafts below: a half-typed key left
 * behind on a pane the user has navigated away from is a key they cannot see and did not
 * save.
 */
function ProviderPane({
  provider,
  models,
  save,
  pending,
}: {
  provider: ProviderSetting;
  models: ProviderModels;
  save: Save;
  pending: boolean;
}) {
  const keyId = useId();
  const regionId = useId();
  const [key, setKey] = useState("");
  const [region, setRegion] = useState("");
  const isBedrock = provider.key === "bedrock";
  const typedKey = key.trim();
  const typedRegion = region.trim();

  const saveKey = () => {
    if (!typedKey) return;
    const patch: SettingsPatch = { provider_keys: { [provider.key]: typedKey } };
    // A key and its region are one edit, so they go in one patch: the engine validates the
    // whole thing or writes none of it.
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

  const rest = models.names.length - NAMED_MODELS;
  return (
    <div className="space-y-5 rounded-lg border bg-card/60 p-4">
      <header className="flex items-center gap-3">
        {/* The mark in a tile of its own, at twice the size the tab draws it. A pane opens
            on a click that was aimed at one row of a list of seven, and this is what says
            it opened on the row the user meant. */}
        <span className="grid size-9 shrink-0 place-items-center rounded-lg border bg-background">
          <ProviderIcon providerKey={provider.key} label={provider.label} className="size-5" />
        </span>
        <div className="min-w-0">
          <h3 className="truncate text-sm font-medium">{provider.label}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {provider.has_key ? `${S.keySaved} · ${provider.key_hint}` : S.noKeyYet}
          </p>
        </div>
      </header>

      <div className="space-y-2">
        <Label htmlFor={keyId}>{S.apiKey}</Label>
        {/* `minmax(0,1fr)` and not `1fr`: an input's automatic minimum size would otherwise
            push the two buttons past the panel's width. */}
        <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2">
          <Input
            id={keyId}
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
            // The draft goes with the stored key: leaving a half-typed one behind in a
            // field whose placeholder has just lost its hint reads as if it were still saved.
            onClick={() => save({ provider_keys: { [provider.key]: null } }, () => setKey(""))}
          >
            {S.clearKey}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">{S.keyWriteOnly}</p>
      </div>

      {isBedrock && (
        <div className="space-y-2">
          <Label htmlFor={regionId}>{S.region}</Label>
          <Input
            id={regionId}
            placeholder={provider.region ?? DEFAULT_BEDROCK_REGION}
            value={region}
            onChange={(e) => setRegion(e.target.value)}
            onBlur={saveRegion}
          />
          <p className="text-xs text-muted-foreground">{S.regionHint}</p>
        </div>
      )}

      {/* What the key is for, which is the one question a key field cannot answer on its
          own. The catalog is the same list the composer picks from, so a provider whose
          models are all greyed out there is explained here. */}
      {models.total > 0 && (
        <div className="space-y-2 border-t pt-4">
          <div className="flex items-baseline gap-2">
            <h4 className="text-xs font-medium">{S.providerModels}</h4>
            {/* The heading already says models, so the count does not: with a key, how
                many are callable; without one, how many are waiting on it. */}
            <p className="text-xs text-muted-foreground">
              {provider.has_key
                ? `${models.available} ${S.modelsAvailable}`
                : `${models.total} ${S.modelsCount} · ${S.modelsNeedKey}`}
            </p>
          </div>
          <div className="flex flex-wrap gap-1">
            {models.names.slice(0, NAMED_MODELS).map((name) => (
              <Badge key={name} variant="outline" className="font-normal text-muted-foreground">
                {name}
              </Badge>
            ))}
            {rest > 0 && (
              <Badge variant="outline" className="font-normal text-muted-foreground">
                +{rest}
              </Badge>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * A number that saves when the field loses focus. The engine happily accepts `0` for
 * either of these, which would end every run before it produced anything, so the floor is
 * 1 here; anything else — blank, a decimal, unchanged — snaps back to the stored value
 * rather than sending a patch.
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

export function SettingsPanel() {
  const qc = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList });
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  const refreshId = useId();
  // Which provider the pane is showing. Derived during render rather than corrected in an
  // effect: the list arrives one query after this component, so the choice is unknown for
  // the first paint and falls back to the first provider until it is made.
  const [chosen, setChosen] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: (patch: SettingsPatch) => api.settingsSet(patch),
    onSuccess: (next) => {
      // `settings_set` answers with the whole new `Settings`, so seed the cache with it and
      // the panel repaints before the refetch lands. The `models` invalidate is not
      // housekeeping: `available` is derived from which keys exist, and it is what takes the
      // no-key banner off `App.tsx` — and fills this provider's model list in — the moment
      // the first key is saved.
      qc.setQueryData(["settings"], next);
      void qc.invalidateQueries({ queryKey: ["settings"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
  const commit: Save = (patch, onDone) => save.mutate(patch, { onSuccess: () => onDone?.() });
  // `variables` is the patch currently in flight, which is enough to name the one pane that
  // should be disabled — no second piece of state to keep in step with the mutation.
  const pendingProvider = save.isPending ? providerOf(save.variables) : null;
  // A mutation and not a bare call, so a failure to open the folder has somewhere to go.
  const openLogs = useMutation({ mutationFn: api.openLogs });

  const s = settings.data;
  const providers = s?.providers ?? [];
  const active = providers.find((p) => p.key === chosen)?.key ?? providers[0]?.key ?? null;

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="px-6 pt-3 pb-4">
        <h1 className="text-xl font-semibold">{S.settings}</h1>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-10">
        <div className="mx-auto max-w-3xl space-y-8">
          <Section title={S.providers}>
            {/* A tab each, down the side rather than across the top: seven providers do not
                fit on one line at this width, and a list that grows downwards is the one a
                new provider can be added to without redrawing the strip. */}
            <Tabs
              orientation="vertical"
              value={active}
              onValueChange={(v) => setChosen(typeof v === "string" ? v : null)}
              className="grid gap-4 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]"
            >
              <TabsList aria-label={S.providers} className="flex-col items-stretch self-start rounded-lg border bg-muted/20 p-1">
                {providers.map((p) => (
                  <TabsTab key={p.key} value={p.key} className="justify-start">
                    <ProviderIcon providerKey={p.key} label={p.label} className="size-4 shrink-0" />
                    <span className="min-w-0 flex-1 truncate text-left">{p.label}</span>
                    {/* Which providers are ready, readable without opening any of them. The
                        word is for a screen reader, which cannot see the dot. */}
                    {p.has_key && (
                      <>
                        <span className="sr-only">{S.keySaved}</span>
                        <span className="size-1.5 shrink-0 rounded-full bg-emerald-500" />
                      </>
                    )}
                  </TabsTab>
                ))}
              </TabsList>
              {providers.map((p) => (
                <TabsPanel key={p.key} value={p.key}>
                  <ProviderPane
                    provider={p}
                    models={modelsOf(models.data, p.label)}
                    save={commit}
                    pending={pendingProvider === p.key}
                  />
                </TabsPanel>
              ))}
            </Tabs>
            {save.isError && <p className="text-sm text-destructive">{api.messageOf(save.error)}</p>}
          </Section>

          {s && (
            <Section title={S.defaultsSection}>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1 sm:col-span-2">
                  <Label>{S.defaultModel}</Label>
                  {/* Unavailable models stay listed but disabled: seeing the model you
                      wanted greyed out is the hint that its provider is missing a key. */}
                  <Select
                    value={s.default_model || null}
                    onValueChange={(m) => {
                      if (m) commit({ default_model: m });
                    }}
                  >
                    <SelectTrigger className="w-full" aria-label={S.defaultModel}>
                      {/* Same reason as `Composer`: Base UI reads the trigger's text from
                          the selected *item*, and the items are in a portal that has not
                          mounted until the list is opened — so the stored default would show
                          as a bare id until then, while the open list showed
                          `provider · name`. */}
                      <SelectValue>
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
                <div className="flex items-center gap-2 sm:col-span-2">
                  {/* Base UI: `onCheckedChange` is `(checked, eventDetails) => void`. */}
                  <Switch
                    id={refreshId}
                    checked={s.catalog_refresh}
                    onCheckedChange={(c) => commit({ catalog_refresh: c })}
                  />
                  <Label htmlFor={refreshId}>{S.catalogRefresh}</Label>
                </div>
              </div>
            </Section>
          )}

          <Section title={S.appSection}>
            <div className="space-y-1 text-xs text-muted-foreground">
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
              {openLogs.isError && <p className="text-sm text-destructive">{api.messageOf(openLogs.error)}</p>}
            </div>
          </Section>
        </div>
      </div>
    </section>
  );
}
