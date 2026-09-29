<script lang="ts">
  import { renderMarkdown } from './markdown';
  import type { TextProps } from './registry';

  let { text }: TextProps = $props();

  // `renderMarkdown` escapes its input before parsing, so the only markup here is the
  // markup it wrote. See the note at the top of `markdown.ts`.
  let html = $derived(renderMarkdown(text));
</script>

<article class="prose">
  {@html html}
</article>

<style>
  .prose {
    max-width: 74ch;
    margin: 0 auto;
    padding: 28px 32px 72px;
    font-size: var(--fs-lg);
    line-height: 1.68;
    color: var(--text);
    overflow-wrap: anywhere;
  }

  /* The document's own tags are written by the renderer, so they are global. */
  .prose :global(h1),
  .prose :global(h2),
  .prose :global(h3),
  .prose :global(h4),
  .prose :global(h5),
  .prose :global(h6) {
    margin: 1.6em 0 0.55em;
    line-height: 1.3;
    font-weight: 650;
    letter-spacing: -0.01em;
  }
  .prose :global(> :first-child) { margin-top: 0; }
  .prose :global(h1) { font-size: var(--fs-3xl); }
  .prose :global(h2) { font-size: var(--fs-2xl); }
  .prose :global(h3) { font-size: var(--fs-xl); }
  .prose :global(h4),
  .prose :global(h5),
  .prose :global(h6) { font-size: var(--fs-lg); }
  .prose :global(h1),
  .prose :global(h2) {
    padding-bottom: 0.3em;
    border-bottom: 1px solid var(--border);
  }

  .prose :global(p) { margin: 0 0 1em; }
  .prose :global(ul),
  .prose :global(ol) { margin: 0 0 1em; padding-left: 1.45em; }
  .prose :global(li) { margin: 0.22em 0; }
  .prose :global(li > ul),
  .prose :global(li > ol) { margin: 0.22em 0 0.35em; }

  .prose :global(a) { color: var(--accent-text); text-decoration: none; }
  .prose :global(a:hover) { text-decoration: underline; }

  .prose :global(strong) { font-weight: 650; }
  .prose :global(del) { color: var(--text-faint); }

  .prose :global(code) {
    padding: 0.12em 0.35em;
    border-radius: 4px;
    background: var(--bg-active);
    font-family: var(--font-mono);
    font-size: 0.875em;
  }
  .prose :global(pre) {
    margin: 0 0 1.2em;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-sunken);
    overflow-x: auto;
  }
  .prose :global(pre code) {
    padding: 0;
    background: none;
    font-size: var(--fs-md);
    line-height: 1.55;
  }

  .prose :global(blockquote) {
    margin: 0 0 1.2em;
    padding: 2px 0 2px 14px;
    border-left: 3px solid var(--border-strong);
    color: var(--text-muted);
  }
  .prose :global(blockquote > :last-child) { margin-bottom: 0; }

  .prose :global(hr) {
    margin: 1.8em 0;
    border: none;
    border-top: 1px solid var(--border);
  }

  .prose :global(table) {
    width: 100%;
    margin: 0 0 1.2em;
    border-collapse: collapse;
    font-size: var(--fs-md);
  }
  .prose :global(th),
  .prose :global(td) {
    padding: 6px 10px;
    border: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
  }
  .prose :global(th) {
    background: var(--bg-sunken);
    font-weight: 600;
    white-space: nowrap;
  }

  .prose :global(img) {
    max-width: 100%;
    border-radius: var(--radius);
  }

  @media (max-width: 640px) {
    .prose { padding: 20px 18px 56px; }
  }
</style>
