// A published brand mark, and the letter that stands in for one that is not published.
//
// Brand marks come from the vendor's own artwork rather than from us: a hand-drawn "close
// enough" logo is both worse at being recognised and a claim about someone else's mark
// that we are not the ones to make. Two sets supply them, and which one a caller reaches
// for is about where the thing being named comes from:
//
//   * **Model providers** take `lib/providerMarks`, which is models.dev's logo set — the
//     same project the model catalog comes from, MIT, one path on one canvas per provider.
//     That set is what makes OpenAI, xAI and Amazon Bedrock drawable at all: `simple-icons`
//     carries none of the three. OpenAI's was in it and was removed, as it is when a brand
//     owner asks; AWS's trademark guidelines allow a plain-text reference to a product name
//     and require prior written approval for the logos and Architecture Icons, so no
//     redistributable set carries an Amazon family.
//   * **Connected sources** take `simple-icons` (Notion), which covers them and is already
//     a dependency.
//
// Either way a mark is its owner's trademark, used to name the provider or the service a
// row *is* — in `currentColor`, at the size of every other row, with nothing about it
// claiming endorsement. MIT covers the file; it is not a trademark licence, and this is the
// one file to revisit if an owner ever asks us to stop.
//
// A provider the marks do not cover gets its initial instead. Not a stand-in glyph: a
// wrench or a sparkle beside real logos reads as the one that failed to load.

/**
 * One mark as an inline SVG. Both sets draw single filled shapes, unlike Lucide's strokes,
 * which is why this sets `fill` and no stroke at all.
 */
export function BrandMark({
  path,
  title,
  className,
  viewBox = "0 0 24 24",
}: {
  path: string;
  title: string;
  className?: string;
  /** The canvas the path was drawn on. `simple-icons` uses 24; models.dev uses 40. */
  viewBox?: string;
}) {
  return (
    <svg viewBox={viewBox} fill="currentColor" className={className} role="img" aria-label={title}>
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
