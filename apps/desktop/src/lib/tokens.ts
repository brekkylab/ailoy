// How a token count is written, wherever one is shown.
//
// One format for all of them — the model picker's context sizes, the usage dialog's totals,
// a rate-limit window — so the same number never reads two ways on one screen. Below 1k a
// count is exact; above it, what matters is the magnitude, and `81,008` is noise next to
// `81k`.

/** `1.05M`, `200k`, `8.2k`, `512`. */
export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${trimZeros((n / 1_000_000).toFixed(2))}M`;
  if (n >= 1000) return `${trimZeros((n / 1000).toFixed(1))}k`;
  return String(n);
}

/** `1.50` → `1.5`, `2.00` → `2`: the zeros a fixed-point format leaves are noise here. */
export function trimZeros(fixed: string): string {
  return fixed.includes(".") ? fixed.replace(/0+$/, "").replace(/\.$/, "") : fixed;
}
