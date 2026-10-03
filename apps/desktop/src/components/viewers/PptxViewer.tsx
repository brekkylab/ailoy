// A PowerPoint deck, laid out as one SVG per slide.
//
// `pptx-viewer-core` parses the archive into a model and exports each slide as SVG with no
// DOM of its own. That shape is why it is the one used here: the same page-list this app
// already gives a PDF and a Hangul document, and text that stays text.
//
// Two things it does not do for itself. Pictures come back as a path into the archive and
// no bytes, so `lib/pptx` fills them in before anything is drawn — see there. And text in a
// table cell is not wrapped, so a long line runs past its cell; that one is the library's,
// and a deck of dense tables shows it.
//
// ~2.3 MB of parser, imported when a deck is opened and not before.

import { useEffect, useRef, useState } from "react";

import type { Presentation } from "pptx-viewer-core";

import { attachMedia } from "@/lib/pptx";
import { report } from "@/lib/report";
import { useBytes } from "@/lib/useBytes";
import { S } from "@/strings";

/** English Metric Units per CSS pixel: the deck's own size is in EMU. */
const EMU_PER_PX = 914400 / 96;

type Slide = Presentation["slides"][number];

/** A deck, opened and ready to draw. */
interface Deck {
  slides: Slide[];
  /** Slide size in pixels, which every slide shares. */
  width: number;
  height: number;
  /** Frees the archive, the caches and every blob URL the pictures came from. */
  dispose: () => void;
}

async function openDeck(bytes: ArrayBuffer): Promise<Deck> {
  const { Presentation: P } = await import("pptx-viewer-core");
  // A copy: the parser keeps what it is given, and `useBytes` holds the original for as
  // long as the file is open.
  const deck = await P.load(bytes.slice(0));
  const slides = deck.slides;
  await attachMedia(slides, (path) => deck.handler.getImageData(path));
  return {
    slides,
    width: deck.width / EMU_PER_PX,
    height: deck.height / EMU_PER_PX,
    dispose: () => deck.dispose(),
  };
}

/** One slide: its own shape until it is on screen, the drawn slide once it has been. */
function SlideView({
  deck,
  index,
  path,
}: {
  deck: Deck;
  index: number;
  path: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = host.current;
    if (!el || visible) return;
    // A margin, so a slide is drawn just before it is scrolled to rather than as it lands.
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { rootMargin: "400px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [visible]);

  useEffect(() => {
    const into = host.current;
    if (!visible || !into) return;
    let live = true;
    void import("pptx-viewer-core")
      .then(({ SvgExporter }) => {
        if (!live || !host.current) return;
        // Markup the exporter generated, not markup carried out of the file — and inline
        // script is refused by the window's CSP either way.
        into.innerHTML = SvgExporter.exportSlide(deck.slides[index], deck.width, deck.height);
        const svg = into.firstElementChild;
        if (svg instanceof SVGSVGElement) {
          // The exporter writes the slide's size in px; the `viewBox` is what should decide
          // the scale, so the column's width can.
          svg.removeAttribute("width");
          svg.removeAttribute("height");
          svg.style.display = "block";
          svg.style.width = "100%";
          svg.style.height = "auto";
        }
      })
      .catch((err: unknown) => {
        report(`pptx slide ${index + 1} of ${path} could not be drawn`, err);
        if (!live || !host.current) return;
        const failed = document.createElement("p");
        failed.className = "p-4 text-xs text-destructive";
        failed.textContent = S.viewerFailed;
        into.replaceChildren(failed);
      });
    return () => {
      live = false;
    };
  }, [visible, deck, index, path]);

  return (
    <div
      ref={host}
      className="bg-card shadow-sm"
      style={{ aspectRatio: `${deck.width} / ${deck.height}` }}
    />
  );
}

export function PptxViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const [deck, setDeck] = useState<Deck | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (bytes.state !== "ready") return;
    let live = true;
    let opened: Deck | null = null;
    void openDeck(bytes.bytes)
      .then((d) => {
        opened = d;
        if (live) setDeck(d);
      })
      .catch((err: unknown) => {
        report(`${path} could not be opened as a presentation`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
      // Every picture on screen is a blob URL this owns; without it they outlive the deck.
      opened?.dispose();
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="p-4 text-xs text-destructive">{S.viewerFailed}</p>;

  return (
    <div className="h-full overflow-auto bg-background px-6 py-4">
      {deck === null ? (
        <p className="text-xs text-muted-foreground">{S.loading}</p>
      ) : (
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          {deck.slides.map((_, i) => (
            <SlideView key={i} deck={deck} index={i} path={path} />
          ))}
        </div>
      )}
    </div>
  );
}
