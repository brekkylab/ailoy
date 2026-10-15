// Filling in the pictures a parsed deck only points at.
//
// `Presentation.load` records where each picture lives inside the archive — `imagePath`,
// e.g. `ppt/media/image1.png` — but leaves `imageData` empty, and the SVG exporter draws a
// grey box labelled "image" for anything with no data. A deck of grey boxes is not a deck,
// so the bytes are fetched before the first slide is drawn.
//
// The runtime that parsed the file still holds the archive and hands back a `blob:` URL per
// media path, which is why this takes a lookup rather than opening the zip a second time.
// Blob URLs also keep the deck out of the SVG: base64 would put a copy of every picture
// into the markup and another into the string it was built from.

/**
 * The shape this needs from a parsed element. The model carries far more, and an element
 * that is a line of text carries none of it — which is why the walk below narrows each one
 * rather than the signature demanding it.
 */
interface MediaElement {
  imagePath?: unknown;
  imageData?: unknown;
  /** A group's members, which hold pictures of their own. */
  children?: unknown;
}

export interface MediaReport {
  /** Pictures that now have their bytes. */
  filled: number;
  /** Pictures whose path the archive had nothing for; these stay as placeholders. */
  missing: number;
}

/**
 * Give every picture in `slides` its data, resolving each path at most once.
 *
 * A path that resolves to nothing is counted and left alone rather than thrown on: one
 * unreadable picture should cost its own box, not the whole deck.
 */
export async function attachMedia(
  slides: readonly { elements?: readonly unknown[] }[],
  resolve: (path: string) => Promise<string | undefined>,
): Promise<MediaReport> {
  const seen = new Map<string, string | undefined>();
  const report: MediaReport = { filled: 0, missing: 0 };

  const walk = async (elements: readonly unknown[] | undefined): Promise<void> => {
    for (const candidate of elements ?? []) {
      if (typeof candidate !== "object" || candidate === null) continue;
      const element = candidate as MediaElement;
      if (Array.isArray(element.children)) await walk(element.children);
      const path = element.imagePath;
      if (typeof path !== "string" || path === "" || element.imageData) continue;
      if (!seen.has(path)) {
        seen.set(path, await resolve(path).catch(() => undefined));
      }
      const data = seen.get(path);
      if (data) {
        element.imageData = data;
        report.filled += 1;
      } else {
        report.missing += 1;
      }
    }
  };

  for (const slide of slides) await walk(slide.elements);
  return report;
}
