// A Hangul word processor document — `.hwp` and `.hwpx` — laid out and drawn as SVG.
//
// `@rhwp/core` is a Rust typesetter compiled to WebAssembly: it reads both the binary HWP
// container and the XML-based HWPX one, paginates the document itself, and hands back one
// SVG per page. Nothing else on npm is close — the alternatives stopped at an alpha in
// 2022 and read only the binary format.
//
// It is ~9.5 MB of WebAssembly, so it is imported the moment a Hangul file is opened and
// not before. The engine is initialised once per window: the cost is paid by the first
// document and no other.
//
// The library asks its host to measure text rather than carrying font metrics of its own,
// which is what `measureTextWidth` below is. It has to exist before a document is parsed:
// line breaking happens during pagination, not during drawing.
//
// Pages are drawn as they are scrolled to, for the reason the PDF viewer does it: a page
// here is one `<text>` element per glyph, so a long document put in the DOM at once is
// tens of thousands of nodes.

import { useEffect, useRef, useState } from "react";

import type { HwpDocument } from "@rhwp/core";

import { report } from "@/lib/report";
import { useBytes } from "@/lib/useBytes";
import { S } from "@/strings";

/** A4 in portrait, for the placeholders before the first page has been measured. */
const DEFAULT_ASPECT = 210 / 297;

let engine: Promise<typeof import("@rhwp/core")> | null = null;
let measureCtx: CanvasRenderingContext2D | null = null;
let measuredFont = "";

/**
 * The text measurement the typesetter calls out for, once per window.
 *
 * A canvas in the document the page is drawn into, so a run is measured in the very font
 * the browser will then paint — measuring anywhere else is how a line comes out a glyph
 * short of where it was broken.
 */
function installMeasure(): void {
  const host = globalThis as { measureTextWidth?: (font: string, text: string) => number };
  if (host.measureTextWidth) return;
  host.measureTextWidth = (font, text) => {
    measureCtx ??= document.createElement("canvas").getContext("2d");
    if (!measureCtx) return 0;
    if (font !== measuredFont) {
      measureCtx.font = font;
      measuredFont = font;
    }
    return measureCtx.measureText(text).width;
  };
}

/**
 * The engine, initialised once.
 *
 * The binary is imported for its URL and fetched from the app's own origin, which is what
 * `default-src 'self'` allows; compiling it is what `'wasm-unsafe-eval'` in `script-src`
 * is there for. Both are in `src-tauri/tauri.conf.json`.
 */
function loadEngine(): Promise<typeof import("@rhwp/core")> {
  engine ??= (async () => {
    const [mod, wasm] = await Promise.all([
      import("@rhwp/core"),
      import("@rhwp/core/rhwp_bg.wasm?url"),
    ]);
    installMeasure();
    await mod.default({ module_or_path: wasm.default });
    return mod;
  })();
  return engine;
}

/** The size the renderer gave a page, off the SVG it produced. */
function aspectOf(svg: string): number | null {
  const box = /viewBox="0 0 ([\d.]+) ([\d.]+)"/.exec(svg);
  if (!box) return null;
  const [, w, h] = box;
  return Number(h) > 0 ? Number(w) / Number(h) : null;
}

/** One page: its own shape until it is on screen, the drawn page once it has been. */
function PageView({
  doc,
  index,
  aspect,
  path,
}: {
  doc: HwpDocument;
  index: number;
  aspect: number;
  path: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = host.current;
    if (!el || visible) return;
    // A margin, so a page is drawn just before it is scrolled to rather than as it lands.
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
    try {
      // Markup the renderer generated, not markup carried out of the file — and inline
      // script is refused by the window's CSP either way, which is what makes this the
      // shape of the insert rather than a hole in it.
      into.innerHTML = doc.renderPageSvg(index);
      const svg = into.firstElementChild;
      if (svg instanceof SVGSVGElement) {
        // The renderer writes the paper's real size in px. The `viewBox` is what should
        // decide the scale, so the column's width can.
        svg.removeAttribute("width");
        svg.removeAttribute("height");
        svg.style.display = "block";
        svg.style.width = "100%";
        svg.style.height = "auto";
      }
    } catch (err) {
      report(`hwp page ${index} of ${path} could not be drawn`, err);
      // Written into the node rather than kept as state: typesetting a page is synchronous,
      // so a state change here would be a second render inside the first one's effect. This
      // element's contents are already this effect's to own.
      const failed = document.createElement("p");
      failed.className = "p-4 text-xs text-destructive";
      failed.textContent = S.viewerFailed;
      into.replaceChildren(failed);
    }
  }, [visible, doc, index, path]);

  return <div ref={host} className="bg-card shadow-sm" style={{ aspectRatio: String(aspect) }} />;
}

export function HwpViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const [doc, setDoc] = useState<HwpDocument | null>(null);
  const [pages, setPages] = useState(0);
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (bytes.state !== "ready") return;
    let live = true;
    let opened: HwpDocument | null = null;
    void loadEngine()
      .then((mod) => {
        const d = new mod.HwpDocument(new Uint8Array(bytes.bytes));
        opened = d;
        // The one page drawn up front, for its shape: every placeholder below is the size
        // of the paper, so the scrollbar is right before anything has been typeset.
        const first = aspectOf(d.renderPageSvg(0));
        if (!live) return;
        if (first) setAspect(first);
        setPages(d.pageCount());
        setDoc(d);
      })
      .catch((err: unknown) => {
        report(`${path} could not be opened as a Hangul document`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
      opened?.free();
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="p-4 text-xs text-destructive">{S.viewerFailed}</p>;

  return (
    <div className="h-full overflow-auto bg-background px-6 py-4">
      {doc === null ? (
        <p className="text-xs text-muted-foreground">{S.loading}</p>
      ) : (
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          {Array.from({ length: pages }, (_, i) => (
            <PageView key={i} doc={doc} index={i} aspect={aspect} path={path} />
          ))}
        </div>
      )}
    </div>
  );
}
