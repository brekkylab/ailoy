// What the model picker shows, worked out without a webview.
//
// The picker is two lists side by side — the vendors, then one vendor's models — with a
// search across all of them on top. Which vendor a model belongs to is matched on the
// provider's label, for the reason `modelsOf` gives in `lib/providers`: `ModelInfo.provider`
// is the label, and the key never reaches a model row.

import { trimZeros } from "@/lib/tokens";
import type { ModelCost, ModelInfo, ProviderSetting } from "@/types";

export interface Vendor {
  key: string;
  label: string;
  hasKey: boolean;
  models: ModelInfo[];
}

/**
 * Every provider the settings list, in the settings' order, each with its models.
 *
 * A provider with no models in the catalog is still a vendor — the catalog may simply not
 * have loaded yet, and a vendor that vanished from the list for that would move every row
 * under it.
 */
export function vendorsOf(providers: ProviderSetting[] | undefined, models: ModelInfo[] | undefined): Vendor[] {
  return (providers ?? []).map((p) => ({
    key: p.key,
    label: p.label,
    hasKey: p.has_key,
    models: (models ?? []).filter((m) => m.provider === p.label),
  }));
}

/**
 * The models a query names, across every vendor.
 *
 * Every word has to appear somewhere in the name, the id or the vendor, so "opus 5" finds
 * `Claude Opus 5` and "bedrock sonnet" finds Bedrock's Sonnets. That is loose on purpose —
 * a version is typed as "5" and has to find "4.5" too when that is what someone means —
 * so the order does the rest. A model the user can call comes before one they cannot; then
 * a name that holds the whole query as typed, so "sonnet 5" leads with `Claude Sonnet 5`
 * and not with the `4.5` its "5" also matched; then a name that *starts* with the first
 * word, so "gpt" does not lead with a model that merely mentions it.
 */
export function searchModels(models: ModelInfo[] | undefined, query: string): ModelInfo[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  const hay = (m: ModelInfo) => `${m.name} ${m.id} ${m.provider}`.toLowerCase();
  const found = (models ?? []).filter((m) => {
    const h = hay(m);
    return words.every((w) => h.includes(w));
  });
  const phrase = words.join(" ");
  const holds = (m: ModelInfo) => m.name.toLowerCase().includes(phrase);
  const leads = (m: ModelInfo) => m.name.toLowerCase().startsWith(words[0]);
  return found
    .map((m, i) => ({ m, i }))
    .sort(
      (a, b) =>
        Number(b.m.available) - Number(a.m.available) ||
        Number(holds(b.m)) - Number(holds(a.m)) ||
        Number(leads(b.m)) - Number(leads(a.m)) ||
        a.i - b.i,
    )
    .map(({ m }) => m);
}

/** The vendor a picker should open on: the one the current model is from, else the first with a key. */
export function initialVendor(vendors: Vendor[], current: string | null): string | null {
  const own = current ? vendors.find((v) => v.models.some((m) => m.id === current)) : undefined;
  return (own ?? vendors.find((v) => v.hasKey) ?? vendors[0])?.key ?? null;
}

/**
 * A model's price as `$in / $out`, per million tokens — the unit models.dev quotes in.
 * `null` when either half is unknown: half a price reads as the whole of one.
 */
export function formatPrice(cost: ModelCost | null): string | null {
  if (cost?.input == null || cost.output == null) return null;
  return `$${trimZeros(cost.input.toFixed(2))} / $${trimZeros(cost.output.toFixed(2))}`;
}
