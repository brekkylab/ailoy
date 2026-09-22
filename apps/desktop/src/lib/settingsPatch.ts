// Reading a settings patch: which part of the pane sent it.
//
// Kept out of the components so the one rule with a decision in it can be exercised
// without a webview.

import type { SettingsPatch } from "@/types";

/** Send a patch; `onDone` runs only if the engine accepted it. */
export type Save = (patch: SettingsPatch, onDone?: () => void) => void;

/**
 * Which provider pane a patch belongs to, or `null` for a patch that is not a pane's —
 * the default model, the turn limits, the catalog switch. One mutation serves a whole
 * page, so this is what keeps a save on one provider from greying out the rest.
 *
 * Bedrock's region and routing are that pane's too: they are sent from the menus under
 * its key, and the engine writes them under the same provider.
 */
export function providerOf(patch: SettingsPatch | undefined): string | null {
  if (!patch) return null;
  const [first] = Object.keys(patch.provider_keys ?? {});
  if (first != null) return first;
  return patch.bedrock_region != null || patch.bedrock_routing != null ? "bedrock" : null;
}
