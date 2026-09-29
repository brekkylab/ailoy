// Where a viewer's file comes from.
//
// The viewers ask a tree four things — a file's address, the folder it sits in, its bytes,
// its characters — so a new tree is a new `Source` and not a second copy of the viewers.
// Today there is one: a context's tree on disk, read through Tauri's asset protocol.

import { convertFileSrc } from "@tauri-apps/api/core";

import { decodeText, type Decoded } from "@/lib/encoding";
import type { Context } from "@/lib/contexts.svelte";
import type { Entry } from "@/lib/viewers/entry";

export interface Source {
  /** The file's address: what a `url` viewer — one the browser renders — is handed. */
  url(entry: Entry): string;
  /**
   * The folder that address sits in, trailing slash and all. A previewed document's
   * relative links resolve against this, so it has to be the file's own folder.
   */
  base(entry: Entry): string;
  /** The file, unchanged — for a viewer that opens a container itself. */
  bytes(entry: Entry): Promise<ArrayBuffer>;
  /** The same read, decoded, with the encoding it was decoded from. */
  decoded(entry: Entry): Promise<Decoded>;
  /** Where this file is, for the line under its name. */
  where(entry: Entry): string;
}

/**
 * The asset-protocol URL of an absolute path, one encoded segment at a time. Not
 * `convertFileSrc(path)` itself: that encodes the slashes too, and a URL with no folders
 * in it gives a document's relative links nothing to resolve against.
 *
 * The root's own slash is the one that stays encoded. Tauri drops the URL path's first
 * character and decodes the rest, so `/%2FUsers/me/a.pdf` reads as `/Users/me/a.pdf`,
 * while `/Users/me/a.pdf` would read as the relative `Users/me/a.pdf` — outside every
 * scope, and refused.
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
