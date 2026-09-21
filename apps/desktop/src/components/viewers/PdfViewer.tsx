// A PDF, drawn by the app rather than by the platform.
//
// The built-in viewer was the obvious choice and it is the one this replaces. It renders
// well, but it paints an opaque backdrop of its own over the whole element, so the band
// around a page is grey in a window that is not — and no CSS reaches it. Measured, not
// assumed: a red background behind a narrow page in an `<iframe>` never appears. Owning the
// rendering is the only way to own what surrounds it.
//
// What that costs is the platform's selection, search and printing, and ~350 KB of
// renderer. The renderer is imported lazily, so only a PDF pays for it.
//
// The bytes come over the bridge, like every other viewer's.
//
// Pages are drawn as they come into view. A report is a hundred pages and a canvas each is
// a hundred bitmaps at device resolution; the placeholders carry each page's real size, so
// the scrollbar is right from the first frame and nothing jumps as pages fill in.

import { useEffect, useRef, useState } from "react";

import { useBytes } from "@/lib/useBytes";
import { report } from "@/lib/report";
import { S } from "@/strings";

/** Page geometry, known before anything is rasterised. */
interface Page {
  index: number;
  width: number;
  height: number;
}

type Task = Awaited<ReturnType<typeof openDocument>>;
type Doc = Awaited<Task["promise"]>;

/**
 * How large a page bitmap may get.
 *
 * WebKit gives up on a canvas past its budget by leaving it blank rather than by throwing,
 * so a page that asks for too much does not fail — it renders as nothing, which is a bug
 * with no symptom to search for. The resolution is capped before it gets there, and a
 * capped page is slightly soft rather than absent.
 */
const MAX_AREA = 12e6;
const MAX_SIDE = 8192;

/**
 * The loading task, not the document: releasing a PDF means destroying the task, which is
 * what tears down the worker with it. The document is on its `promise`.
 */
async function openDocument(bytes: ArrayBuffer) {
  const pdfjs = await import("pdfjs-dist");
  // The worker is bundled from our own origin, which is what `script-src 'self'` allows.
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url,
  ).toString();
  // The buffer is transferred to the worker, so it must not be the one `useBytes` holds —
  // a re-render would hand over an already-detached buffer.
  return pdfjs.getDocument({ data: bytes.slice(0) });
}

/** One page: its own size until it is on screen, a canvas once it has been. */
function PageView({ doc, page, width, path }: { doc: Doc; page: Page; width: number; path: string }) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [failed, setFailed] = useState(false);

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
    if (!visible || !into || width <= 0) return;
    let live = true;
    let task: { cancel: () => void } | null = null;

    void doc
      .getPage(page.index)
      .then((p) => {
        if (!live) return;
        const scale = width / page.width;
        // Device resolution for a sharp page, held under what the engine will actually
        // rasterise. `getViewport` twice rather than arithmetic on the first: a page's own
        // rotation is in there, and guessing at it is how a landscape page comes out the
        // wrong way round.
        const css = p.getViewport({ scale });
        const budget = Math.min(
          Math.sqrt(MAX_AREA / (css.width * css.height)),
          MAX_SIDE / css.width,
          MAX_SIDE / css.height,
        );
        const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, budget));
        const viewport = p.getViewport({ scale: scale * ratio });

        const canvas = document.createElement("canvas");
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = "100%";
        canvas.style.display = "block";
        // In the tree before it is drawn, so a page appears as it fills rather than after.
        into.replaceChildren(canvas);

        // `canvas` alone. Handing over a `canvasContext` as well is the backwards-compatible
        // form, and pdf.js is explicit that the canvas must then be null — passing both is
        // neither shape, and what comes back is a canvas nothing was painted into.
        const render = p.render({ canvas, viewport });
        task = render;
        return render.promise;
      })
      .catch((err: unknown) => {
        // A cancel is not a failure: this effect re-runs on a width change and cancels the
        // render it replaces.
        if (!live) return;
        report(`pdf page ${page.index} of ${path} could not be drawn`, err);
        setFailed(true);
      });

    return () => {
      live = false;
      task?.cancel();
    };
  }, [visible, doc, page.index, page.width, width, path]);

  return (
    <div
      ref={host}
      className="bg-card shadow-sm"
      // The page's own aspect, so the column is the right height before it is drawn.
      style={{ aspectRatio: `${page.width} / ${page.height}` }}
    >
      {failed && <p className="p-4 text-xs text-destructive">{S.viewerFailed}</p>}
    </div>
  );
}

export function PdfViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const column = useRef<HTMLDivElement>(null);
  const [doc, setDoc] = useState<Doc | null>(null);
  const [pages, setPages] = useState<Page[]>([]);
  const [failed, setFailed] = useState(false);
  const [width, setWidth] = useState(0);

  // The column's width decides the render scale, and it changes with the window and with
  // the sidebar being collapsed.
  useEffect(() => {
    const el = column.current;
    if (!el) return;
    // Rounded, so a sub-pixel reflow does not re-rasterise every page on screen.
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.round(entry.contentRect.width)),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (bytes.state !== "ready") return;
    let live = true;
    let opened: Task | null = null;
    void openDocument(bytes.bytes)
      .then(async (task) => {
        opened = task;
        const d = await task.promise;
        // Geometry for every page up front: it is metadata, not pixels, and it is what
        // lets the placeholders below be the right size.
        const read: Page[] = [];
        for (let i = 1; i <= d.numPages; i += 1) {
          const p = await d.getPage(i);
          const v = p.getViewport({ scale: 1 });
          read.push({ index: i, width: v.width, height: v.height });
        }
        if (!live) return;
        setDoc(d);
        setPages(read);
      })
      .catch((err: unknown) => {
        // What a reader says about a malformed document names parts of a file format,
        // which is not what belongs on screen. The pane says the one actionable thing.
        report(`${path} could not be opened as a pdf`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
      void opened?.destroy();
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="p-4 text-xs text-destructive">{S.viewerFailed}</p>;

  // The width a page is laid out at: the column less its padding, never negative — the
  // first render has no measurement yet, and a negative width used to reach `PageView` as
  // a scale of its own.
  const pageWidth = Math.max(0, Math.min(width - 48, 896));

  return (
    <div ref={column} className="h-full overflow-auto bg-background px-6 py-4">
      {doc === null ? (
        <p className="text-xs text-muted-foreground">{S.loading}</p>
      ) : (
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          {pages.map((page) => (
            <PageView key={page.index} doc={doc} page={page} width={pageWidth} path={path} />
          ))}
        </div>
      )}
    </div>
  );
}
