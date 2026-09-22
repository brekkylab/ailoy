// A published brand mark, and the letter that stands in for one that is not published.
//
// Brand marks come from the vendor rather than from us wherever one is actually published
// for reuse: a hand-drawn "close enough" logo is both worse at being recognised and a
// claim about someone else's mark that we are not the ones to make. `simple-icons` is
// where the published ones live — the vendors' own artwork, CC0, kept up to date as brands
// change — and what it does not carry, it does not carry for a reason. AWS's trademark
// terms forbid redistributing their marks in an icon set; OpenAI's and xAI's are absent
// the same way.
//
// So the alternative is not a drawing of the mark. It is a letter, set in the app's own
// type on the same canvas as the marks, which claims nothing and still tells the rows of a
// list apart. Both render in `currentColor` so a column of them reads as one column rather
// than as a row of stickers; `simple-icons` carries the official hex on each icon if that
// is ever wanted.

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
 * One letter on the marks' canvas. An SVG and not a styled `span` so that it takes a
 * `size-*` class and lines up with them to the pixel, at every size, without a second set
 * of rules for the box around it.
 */
export function LetterMark({ letter, title, className }: { letter: string; title: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} role="img" aria-label={title}>
      {/* `central` and not `middle`: the latter centres on the baseline-to-x-height box and
          sits a letter visibly high in a square this small. */}
      <text
        x="12"
        y="12"
        textAnchor="middle"
        dominantBaseline="central"
        fill="currentColor"
        fontSize="20"
        fontWeight="700"
      >
        {letter}
      </text>
    </svg>
  );
}
