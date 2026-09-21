// What each connected source is drawn as.
//
// Brand marks come from the vendor rather than from us wherever one is actually published
// for reuse: a hand-drawn "close enough" logo is both worse at being recognised and a
// claim about someone else's mark that we are not the ones to make.
//
// `simple-icons` is where the published ones live — the vendors' own artwork, CC0, kept up
// to date as brands change. Notion is in it. Amazon is not, and that is deliberate on
// their part rather than a gap: AWS's trademark terms do not permit their marks to be
// redistributed in an icon set, so simple-icons carries no Amazon family at all and no
// other redistributable source has them either. S3 therefore gets a neutral storage glyph
// from Lucide, at the same weight as the rest. If the team decides to accept AWS's terms
// for their Architecture Icons, this is the one function that changes.
//
// The marks render in `currentColor`, not their brand colours, so the list reads as one
// column rather than as a row of stickers. `simple-icons` carries the official hex on each
// icon if that is ever wanted.

import { Database, Folder, HardDrive } from "lucide-react";
import { siNotion } from "simple-icons";

import type { MountKind } from "@/types";

/**
 * A `simple-icons` entry as an inline SVG. Their paths are single filled shapes on a 24x24
 * canvas, unlike Lucide's strokes, which is why this sets `fill` and no stroke at all.
 */
function BrandMark({ path, title, className }: { path: string; title: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} role="img" aria-label={title}>
      <path d={path} />
    </svg>
  );
}

/**
 * The glyph for a source. `root` is the workspace itself — a local mount like any other, so
 * it is not a kind, but it is the machine rather than a folder on it and takes the disk.
 */
export function SourceIcon({
  kind,
  root,
  className,
}: {
  kind: MountKind;
  root?: boolean;
  className?: string;
}) {
  if (root) return <HardDrive className={className} aria-hidden="true" />;
  switch (kind) {
    case "notion":
      return <BrandMark path={siNotion.path} title={siNotion.title} className={className} />;
    case "s3":
      return <Database className={className} aria-hidden="true" />;
    case "local":
      return <Folder className={className} aria-hidden="true" />;
  }
}
