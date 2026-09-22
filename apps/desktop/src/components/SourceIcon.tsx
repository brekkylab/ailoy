// What each connected source is drawn as.
//
// Notion has a published mark and gets it; see `BrandMark` for when a vendor's does not
// come from the vendor. Amazon's absence there is deliberate on AWS's part rather than a
// gap in `simple-icons`, so S3 takes a neutral storage glyph from Lucide at the same weight
// as the rest. If the team ever accepts AWS's terms for their Architecture Icons, this is
// the one function that changes.

import { Database, Folder, HardDrive } from "lucide-react";
import { siNotion } from "simple-icons";

import { BrandMark } from "@/components/BrandMark";
import type { MountKind } from "@/types";

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
