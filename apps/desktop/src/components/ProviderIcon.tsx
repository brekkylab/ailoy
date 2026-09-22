// What each model provider is drawn as.
//
// The table is `lib/providerMarks`, keyed on the engine's provider key
// (`core/src/providers.rs`). A key missing from it is not a bug — it falls back to the
// provider's initial, which is what a provider the engine learns about before we have its
// mark will get. See `BrandMark` for where the marks come from and on what terms.

import { BrandMark, LetterMark } from "@/components/BrandMark";
import { monogram } from "@/lib/providers";
import { MARK_VIEW_BOX, PROVIDER_MARKS } from "@/lib/providerMarks";

export function ProviderIcon({
  providerKey,
  label,
  className,
}: {
  /** The engine's key for the provider, e.g. `anthropic`. */
  providerKey: string;
  /** Its display name, which names the icon and supplies the fallback letter. */
  label: string;
  className?: string;
}) {
  const path = PROVIDER_MARKS[providerKey];
  if (path) return <BrandMark path={path} title={label} viewBox={MARK_VIEW_BOX} className={className} />;
  return <LetterMark letter={monogram(label)} title={label} className={className} />;
}
