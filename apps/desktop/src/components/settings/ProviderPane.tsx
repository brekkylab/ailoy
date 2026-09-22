// One provider's pane, behind its tab on the Models page.
//
// API keys are write-only from the webview's point of view. They are typed into a
// `type="password"` field, held in component state only while the row is being edited,
// handed to `settings_set` once, and then dropped; the engine never hands one back — a
// saved key comes home as `has_key` plus a `key_hint` suffix, and that is all this file
// ever renders.

import { useId, useState } from "react";

import { ProviderIcon } from "@/components/ProviderIcon";
import { Choice } from "@/components/settings/fields";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ProviderModels } from "@/lib/providers";
import type { Save } from "@/lib/settingsPatch";
import { S } from "@/strings";
import type { ProviderSetting } from "@/types";

/** How many model names a provider's pane lists before it counts the rest instead. */
const NAMED_MODELS = 8;

/**
 * One provider's settings: the key it is reached with, where the call goes, and what that
 * key unlocks.
 *
 * Where it goes is two menus, and they are not the same question. The *region* is the
 * endpoint — `bedrock-runtime.<region>.amazonaws.com`, one host, one set of credentials.
 * The *routing* is the inference profile in front of the model: Bedrock offers the same
 * model as `global.…` for dynamic routing and as `us.…`/`eu.…` for guaranteed data
 * routing, and it is what decides which id the picker offers (see
 * `catalog::fold_region_profiles`). Both are menus rather than fields because both are
 * closed sets the engine already knows — the regions from ailoy, the profiles from the
 * catalog — and a typo in either is a run that fails much later.
 *
 * Unmounted with its tab, which is what resets the key draft: a half-typed key left behind
 * on a pane the user has navigated away from is a key they cannot see and did not save.
 */
export function ProviderPane({
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
  const [key, setKey] = useState("");
  const typedKey = key.trim();

  const saveKey = () => {
    if (!typedKey) return;
    save({ provider_keys: { [provider.key]: typedKey } }, () => setKey(""));
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

      {/* Both gated on the engine having somewhere to send the call *to* rather than on the
          provider's name, so the next provider with regions needs nothing here. */}
      {provider.regions.length > 0 && (
        <Choice
          label={S.region}
          hint={S.regionHint}
          value={provider.region}
          options={provider.regions.map((r) => ({ id: r, label: r }))}
          disabled={pending}
          onPick={(r) => save({ bedrock_region: r })}
        />
      )}
      {provider.routings.length > 0 && (
        <Choice
          label={S.routing}
          hint={S.routingHint}
          value={provider.routing}
          options={provider.routings}
          disabled={pending}
          onPick={(r) => save({ bedrock_routing: r })}
        />
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
