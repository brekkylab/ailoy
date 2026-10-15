// Reading the provider list the settings panel draws.
//
// Which providers exist, and what each is called, is the engine's `PROVIDERS` table
// (`core/src/providers.rs`) — never a copy of it here. These are the two derivations the
// panel needs on top of it, kept out of the component so they can be exercised without a
// webview.

import type { ModelInfo } from "@/types";

/**
 * The letter a provider with no published brand mark is drawn as.
 *
 * The *last* word of the label, because that is the distinctive half of a vendor-plus-
 * product name: Amazon **B**edrock, not Amazon. One letter, so it sits in the same square
 * as a real mark at every size the panel uses.
 */
export function monogram(label: string): string {
  const words = label.trim().split(/\s+/);
  return (words[words.length - 1] ?? label).charAt(0).toUpperCase();
}

/** What a provider's tab can say about the models the key unlocks. */
export interface ProviderModels {
  total: number;
  available: number;
  /** Their names, in the order the catalog gave them: available first, then by id. */
  names: string[];
}

/**
 * The catalog entries that belong to one provider.
 *
 * Matched on the **label** and not the key: `models_list` fills `ModelInfo.provider` with
 * `ProviderDef.label` (`core/src/engine.rs`), which is the same string `ProviderSetting`
 * carries, while the keys it is built from ("xai") and the model-id prefixes ("x-ai/") are
 * a third spelling that never reaches the webview.
 */
export function modelsOf(models: ModelInfo[] | undefined, label: string): ProviderModels {
  const mine = (models ?? []).filter((m) => m.provider === label);
  return {
    total: mine.length,
    available: mine.filter((m) => m.available).length,
    names: mine.map((m) => m.name),
  };
}
