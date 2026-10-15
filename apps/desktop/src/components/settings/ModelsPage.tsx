// The models page: which providers this install can call, and what a new run inherits.
//
// Every write goes through the one `save` mutation so that the engine's validation — which
// happens *before* anything is written, so a rejected patch changes nothing — has a single
// place to land (`api.messageOf`, under the section that sent it).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useState } from "react";

import * as api from "@/api";
import { ProviderIcon } from "@/components/icons/ProviderIcon";
import { NumberSetting, Section } from "@/components/settings/fields";
import { ProviderPane } from "@/components/settings/ProviderPane";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { catalogQuery, useRefreshModels } from "@/lib/catalog";
import { modelsOf } from "@/lib/providers";
import { providerOf, type Save } from "@/lib/settingsPatch";
import { formatRelativeTime } from "@/lib/time";
import { S } from "@/strings";
import type { SettingsPatch } from "@/types";

export function ModelsPage() {
  const qc = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList });
  const catalog = useQuery(catalogQuery);
  const refreshModels = useRefreshModels();
  const fetching = refreshModels.isPending || !!catalog.data?.refreshing;
  // The clock "updated 3h ago" is read against, ticking like the sidebar's.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);
  const refreshId = useId();
  // Which provider the pane is showing. Derived during render rather than corrected in an
  // effect: the list arrives one query after this component, so the choice is unknown for
  // the first paint and falls back to the first provider until it is made.
  const [chosen, setChosen] = useState<string | null>(null);

  const save = useMutation({
    mutationFn: (patch: SettingsPatch) => api.settingsSet(patch),
    onSuccess: (next) => {
      // `settings_set` answers with the whole new `Settings`, so seed the cache with it and
      // the page repaints before the refetch lands. The `models` invalidate is not
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

  const s = settings.data;
  const providers = s?.providers ?? [];
  const active = providers.find((p) => p.key === chosen)?.key ?? providers[0]?.key ?? null;

  return (
    <div className="space-y-8">
      <Section title={S.providers}>
        {/* A tab each, down the side rather than across the top: seven providers do not fit
            on one line at this width, and a list that grows downwards is the one a new
            provider can be added to without redrawing the strip. Across the top is where
            the *pages* are, and two strips in the same direction would read as one. */}
        <Tabs
          orientation="vertical"
          value={active}
          onValueChange={(v) => setChosen(typeof v === "string" ? v : null)}
          className="grid gap-4 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]"
        >
          <TabsList
            aria-label={S.providers}
            className="flex-col items-stretch self-start rounded-lg border bg-muted/20 p-1"
          >
            {providers.map((p) => (
              <TabsTab key={p.key} value={p.key} className="justify-start">
                <ProviderIcon providerKey={p.key} label={p.label} className="size-4 shrink-0" />
                <span className="min-w-0 flex-1 truncate text-left">{p.label}</span>
                {/* Which providers are ready, readable without opening any of them. The word
                    is for a screen reader, which cannot see the dot. */}
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
            {/* Keyed on the stored value: when the engine answers with a different one the
                field remounts and re-seeds its draft from it. */}
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
            {/* How old the list is, so a model missing from it reads as "not fetched since"
                rather than "does not exist" — and the way to fetch it, whether or not the
                switch above is on. */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground sm:col-span-2">
              <span>
                {catalog.data?.fetched_at == null
                  ? S.catalogNever
                  : `${S.catalogUpdated} ${formatRelativeTime(catalog.data.fetched_at, now)} · ${catalog.data.models} ${S.modelsCount}`}
              </span>
              <Button size="sm" variant="outline" onClick={() => refreshModels.mutate()} disabled={fetching}>
                {fetching ? S.refreshing : S.refreshNow}
              </Button>
              {catalog.data?.error && !fetching && (
                <span className="text-destructive" title={catalog.data.error}>
                  {S.modelsLoadFailed}
                </span>
              )}
            </div>
          </div>
        </Section>
      )}
    </div>
  );
}
