// Reading the settings the rest of the app branches on.
//
// Kept out of `api.ts` so it may be exercised without `@tauri-apps/api`, which needs a
// webview.

import type { Settings } from "@/types";

/**
 * Whether any provider has a key stored. Settings that have not loaded yet read as "yes":
 * the no-key banner and the composer's placeholder are corrections to a mistake the user
 * has already made, and flashing them at every cold start would be a lie for as long as
 * the first query is in flight.
 */
export function hasAnyKey(settings: Settings | undefined | null): boolean {
  return settings == null || settings.providers.some((p) => p.has_key);
}
