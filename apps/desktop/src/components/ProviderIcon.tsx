// What each model provider is drawn as.
//
// The table is keyed on the engine's provider key (`core/src/providers.rs`), and a key
// missing from it is not a bug: it falls back to the provider's initial, which is what
// every provider whose mark is not publishable for reuse gets. See `BrandMark` for why
// that is a letter and not a drawing of the mark.

import { siAnthropic, siDeepseek, siGooglegemini, siMoonshotai } from "simple-icons";

import { BrandMark, LetterMark } from "@/components/BrandMark";
import { monogram } from "@/lib/providers";

const MARKS: Record<string, { path: string; title: string }> = {
  anthropic: siAnthropic,
  google: siGooglegemini,
  deepseek: siDeepseek,
  moonshotai: siMoonshotai,
};

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
  const mark = MARKS[providerKey];
  if (mark) return <BrandMark path={mark.path} title={label} className={className} />;
  return <LetterMark letter={monogram(label)} title={label} className={className} />;
}
