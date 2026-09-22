// A published brand mark, and the letter that stands in for one that is not published.
//
// Brand marks come from the vendor rather than from us wherever one is actually published
// for reuse: a hand-drawn "close enough" logo is both worse at being recognised and a
// claim about someone else's mark that we are not the ones to make. `simple-icons` is
// where the published ones live — the vendors' own artwork, CC0, kept up to date as brands
// change — and what it does not carry, it does not carry for a reason:
//
//   * **OpenAI** was in it and is not any more. Removals happen when the brand owner asks
//     for one (`removals@simpleicons.org`), and theirs went in a major release. Their
//     developer terms license the OpenAI assets "solely to promote your apps", per brand
//     guidelines we would have to be granted rather than assume — not to label a row in
//     someone else's settings pane.
//   * **Amazon Bedrock**: AWS's trademark guidelines allow a plain-text nominative
//     reference ("works with Amazon Bedrock") and require prior written approval for the
//     logos and the Architecture Icons. There is no version of vendoring their mark here
//     that those guidelines permit, which is why `simple-icons` carries no Amazon family
//     at all.
//   * **xAI**: simply absent, with no published set to take it from.
//
// So the alternative is not a drawing of the mark, and it is not a stand-in glyph either —
// a wrench or a sparkle next to three real logos reads as the one that failed to load. It
// is a letter on a plate, set in the app's own type on the marks' canvas, which claims
// nothing and still tells the rows of a list apart. Everything renders in `currentColor`
// so a column of them reads as one column rather than as a row of stickers; `simple-icons`
// carries the official hex on each icon if that is ever wanted.
//
// If the team would rather ship the real marks, that is a decision about someone else's
// trademark and not a missing feature: `ProviderIcon`'s table is where it lands.

/**
 * A `simple-icons` entry as an inline SVG. Their paths are single filled shapes on a 24x24
 * canvas, unlike Lucide's strokes, which is why this sets `fill` and no stroke at all.
 */
export function BrandMark({ path, title, className }: { path: string; title: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} role="img" aria-label={title}>
      <path d={path} />
    </svg>
  );
}

/**
 * A provider's initial, on the marks' canvas.
 *
 * An SVG and not a styled `span` so that it takes a `size-*` class and lines up with the
 * marks to the pixel, at every size, without a second set of rules for the box around it.
 * The plate behind the letter is what keeps it from reading as a mark that failed to load:
 * a filled shape at the size the others are, which is what the eye compares.
 */
export function LetterMark({ letter, title, className }: { letter: string; title: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} role="img" aria-label={title}>
      <rect x="1.5" y="1.5" width="21" height="21" rx="5.5" fill="currentColor" opacity="0.16" />
      {/* `central` and not `middle`: the latter centres on the baseline-to-x-height box and
          sits a letter visibly high in a square this small. */}
      <text
        x="12"
        y="12.4"
        textAnchor="middle"
        dominantBaseline="central"
        fill="currentColor"
        fontSize="14"
        fontWeight="700"
      >
        {letter}
      </text>
    </svg>
  );
}
