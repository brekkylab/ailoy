// Where a viewer's file comes from.
//
// Viewers ask a tree four things (a file's URL, its folder, its bytes, its text), so a new
// tree is a new `Source`, not a copy of the viewers. The one implementation is a context's
// tree on disk, read through Tauri's asset protocol.

import { convertFileSrc } from "@tauri-apps/api/core";

import { decodeText, type Decoded } from "@/lib/encoding";
import type { Context } from "@/lib/contexts.svelte";
import type { Entry } from "@/lib/viewers/entry";

export interface Source {
  /** The file's URL, handed to `url` viewers. */
  url(entry: Entry): string;
  /**
   * The folder URL, with trailing slash. A previewed document's relative links resolve
   * against it, so it must be the file's own folder.
   */
  base(entry: Entry): string;
  /** The raw bytes, for viewers that open containers themselves. */
  bytes(entry: Entry): Promise<ArrayBuffer>;
  /** The same bytes decoded, with the encoding used. */
  decoded(entry: Entry): Promise<Decoded>;
  /** Location shown under the file name. */
  where(entry: Entry): string;
}

/**
 * The asset-protocol URL of an absolute path, encoded per segment. Not
 * `convertFileSrc(path)` alone: it encodes the slashes too, leaving a document's relative
 * links nothing to resolve against.
 *
 * The root slash stays encoded. Tauri drops the path's first character and decodes the
 * rest, so `/%2FUsers/me/a.pdf` reads as `/Users/me/a.pdf`, whereas `/Users/me/a.pdf`
 * would read as relative `Users/me/a.pdf`, outside every scope and refused.
 */
function assetUrl(parts: string[]): string {
  return convertFileSrc("") + "%2F" + parts.filter(Boolean).map(encodeURIComponent).join("/");
}

/** A context's tree, as the Files view shows it. */
export function contextSource(context: Context): Source {
  const root = context.dir.split("/");
  const url = (entry: Entry) => assetUrl([...root, ...entry.segments]);
  const bytes = async (entry: Entry) => {
    const res = await fetch(url(entry));
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.arrayBuffer();
  };
  return {
    url,
    base: (entry) => assetUrl([...root, ...entry.segments.slice(0, -1)]) + "/",
    bytes,
    decoded: async (entry) => decodeText(await bytes(entry)),
    where: (entry) => [context.name, ...entry.segments.slice(0, -1)].join(" / "),
  };
}
