<script lang="ts">
  import type { PDFDocumentLoadingTask, PDFDocumentProxy } from 'pdfjs-dist/legacy/build/pdf.mjs';
  import { tick } from 'svelte';
  import './pdf-text-layer.css';
  import type { BytesProps } from './registry';

  let { entry, bytes }: BytesProps = $props();

  // Drawn by pdf.js into canvases, not the browser's viewer in an iframe: WKWebView draws
  // nothing for a PDF in a subframe served over a custom scheme, which is how context files
  // are served. The legacy build runs on older macOS WebKit.

  /// What pdf.js fetches by URL (see `pdfjsAssets` in `vite.config.ts`). Absolute, because
  /// the worker resolves them with no page to be relative to.
  const assets = (dir: string) => new URL(`/pdfjs/${dir}/`, location.href).href;

  type Pdfjs = typeof import('pdfjs-dist/legacy/build/pdf.mjs');

  let host: HTMLDivElement;
  /// Each page's box in CSS pixels, and the scale from the PDF's own units to it.
  let pages = $state<{ width: number; height: number; scale: number }[]>([]);
  let failed = $state<string | null>(null);

  $effect(() => {
    let task: PDFDocumentLoadingTask | null = null;
    let doc: PDFDocumentProxy | null = null;
    let observer: IntersectionObserver | null = null;
    let cancelled = false;

    (async () => {
      try {
        const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');
        const worker = await import('pdfjs-dist/legacy/build/pdf.worker.min.mjs?url');
        pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
        // A copy: pdf.js hands the buffer to its worker, which detaches it here.
        task = pdfjs.getDocument({
          data: new Uint8Array(bytes.slice(0)),
          cMapUrl: assets('cmaps'),
          standardFontDataUrl: assets('standard_fonts'),
          iccUrl: assets('iccs'),
          wasmUrl: assets('wasm'),
        });
        doc = await task.promise;
        if (cancelled) return;

        // Size every page box first, at the panel's width, so the scrollbar reflects the full
        // document before any page is drawn.
        const fit = Math.max(200, host.clientWidth - 48);
        const sizes = [];
        for (let n = 1; n <= doc.numPages; n++) {
          const page = await doc.getPage(n);
          const base = page.getViewport({ scale: 1 });
          const scale = fit / base.width;
          sizes.push({ width: base.width * scale, height: base.height * scale, scale });
        }
        if (cancelled) return;
        pages = sizes;

        // Then draw each page once, as it nears the viewport.
        const drawn = new Set<number>();
        observer = new IntersectionObserver(
          (seen) => {
            for (const item of seen) {
              const n = Number((item.target as HTMLElement).dataset.page);
              if (!item.isIntersecting || drawn.has(n) || !doc) continue;
              drawn.add(n);
              void draw(pdfjs, doc, n, item.target as HTMLElement);
            }
          },
          { root: host, rootMargin: '600px 0px' },
        );
        // The page boxes exist once the DOM has caught up with `pages`.
        await tick();
        for (const el of host.querySelectorAll('[data-page]')) observer.observe(el);
      } catch (e) {
        if (!cancelled) failed = e instanceof Error ? e.message : String(e);
      }
    })();

    return () => {
      cancelled = true;
      observer?.disconnect();
      // Destroy the task, not the document: it takes the document and its worker with it.
      void task?.destroy();
    };
  });

  /// Draws page `n` into its box: the canvas image, plus a transparent text layer over each
  /// glyph so text can be selected and copied.
  async function draw(pdfjs: Pdfjs, doc: PDFDocumentProxy, n: number, box: HTMLElement) {
    const page = await doc.getPage(n);
    const { scale } = pages[n - 1];
    const viewport = page.getViewport({ scale });

    // Backing store in device pixels, box in CSS pixels, so it is sharp on Retina.
    const ratio = window.devicePixelRatio || 1;
    const canvas = box.querySelector('canvas')!;
    canvas.width = Math.floor(viewport.width * ratio);
    canvas.height = Math.floor(viewport.height * ratio);
    await page.render({ canvas, viewport: page.getViewport({ scale: scale * ratio }) }).promise;

    const text = box.querySelector<HTMLElement>('.textLayer')!;
    await new pdfjs.TextLayer({
      textContentSource: page.streamTextContent(),
      container: text,
      viewport,
    }).render();
  }
</script>

<div class="pdf" bind:this={host} aria-label={entry.name}>
  {#if failed}
    <p class="note">That PDF could not be opened: {failed}</p>
  {:else if pages.length === 0}
    <p class="note">Loading…</p>
  {/if}
  {#each pages as size, i (i)}
    <div
      class="page"
      data-page={i + 1}
      style:width="{size.width}px"
      style:height="{size.height}px"
      style:--total-scale-factor={size.scale}
    >
      <canvas></canvas>
      <div class="textLayer"></div>
    </div>
  {/each}
</div>

<style>
  .pdf {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    height: 100%;
    padding: 24px;
    overflow: auto;
    /* Letterboxing around the page, distinct from the panel background. */
    background: var(--bg-sunken);
    box-sizing: border-box;
  }
  .page {
    position: relative;
    flex: none;
    background: white;
    box-shadow: var(--shadow-md);
  }
  canvas { display: block; width: 100%; height: 100%; }
  .note { margin: 40px 0; font-size: var(--fs-md); color: var(--text-faint); }
</style>
