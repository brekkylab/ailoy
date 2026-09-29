<script lang="ts">
  import { extOf } from '../files';
  import type { UrlProps } from './registry';

  let { entry, url }: UrlProps = $props();

  /// `loading` until the browser has decoded the bytes. `failed` covers a file named `.png`
  /// that is not one; the panel says so instead of showing a broken-image glyph.
  let status = $state<'loading' | 'ready' | 'failed'>('loading');
  /// `fit` scales an oversized image down to the panel; `actual` is one image pixel per CSS
  /// pixel, needed to read a screenshot.
  let zoom = $state<'fit' | 'actual'>('fit');
  let natural = $state<{ w: number; h: number } | null>(null);

  /// The panel's size, to tell whether the image exceeds it; only then is there anything to
  /// zoom.
  let frame = $state({ w: 0, h: 0 });

  // The viewer is reused across files, so the old image's state must not outlive it.
  $effect(() => {
    void url;
    status = 'loading';
    zoom = 'fit';
    natural = null;
  });

  let oversize = $derived(
    natural !== null && frame.w > 0 && (natural.w > frame.w || natural.h > frame.h),
  );

  function onLoad(event: Event) {
    const img = event.currentTarget as HTMLImageElement;
    // An SVG without width/height has no intrinsic size; the browser's fallback says nothing
    // about the file, so it is not shown.
    natural = img.naturalWidth ? { w: img.naturalWidth, h: img.naturalHeight } : null;
    status = 'ready';
  }
</script>

<div
  class="stage"
  class:scroll={zoom === 'actual'}
  bind:clientWidth={frame.w}
  bind:clientHeight={frame.h}
>
  {#if status === 'failed'}
    <p class="note">That image could not be decoded.</p>
  {:else}
    <!--
      A plain <img>, including for SVG: an image context runs no script and loads no external
      references, so the one type the server marks as active content cannot execute here (see
      `is_active_content` on the server).
    -->
    <img
      src={url}
      alt={entry.name}
      class:hidden={status === 'loading'}
      class:actual={zoom === 'actual'}
      decoding="async"
      draggable="false"
      onload={onLoad}
      onerror={() => (status = 'failed')}
    />
    {#if status === 'loading'}
      <p class="note">Loading…</p>
    {/if}
  {/if}
</div>

<footer>
  <span>
    {#if natural}
      {natural.w.toLocaleString()} × {natural.h.toLocaleString()}
      <span class="dot">·</span>
    {/if}
    {(extOf(entry.name) || 'image').toUpperCase()}
  </span>
  {#if oversize}
    <button onclick={() => (zoom = zoom === 'fit' ? 'actual' : 'fit')}>
      {zoom === 'fit' ? 'Actual size' : 'Fit to panel'}
    </button>
  {/if}
</footer>

<style>
  .stage {
    display: grid;
    place-items: center;
    /* Leaves room for the 30px footer. The panel behind scrolls; the stage fills the rest. */
    height: calc(100% - 30px);
    padding: 16px;
    /* The checkerboard makes transparency read as transparent, not as the theme background. */
    background-color: var(--bg-sunken);
    background-image:
      linear-gradient(45deg, var(--border) 25%, transparent 25%),
      linear-gradient(-45deg, var(--border) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, var(--border) 75%),
      linear-gradient(-45deg, transparent 75%, var(--border) 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0;
  }
  .stage.scroll {
    /* At actual size the image sizes the box, and the box scrolls. */
    place-items: start;
    overflow: auto;
  }

  img {
    display: block;
    max-width: 100%;
    max-height: 100%;
    /* `max-` only scales down, so a 16px icon stays 16px. */
    box-shadow: var(--shadow-md);
  }
  img.actual {
    max-width: none;
    max-height: none;
  }
  .hidden {
    display: none;
  }

  .note {
    margin: 0;
    font-size: var(--fs-md);
    color: var(--text-faint);
  }

  footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    height: 30px;
    padding: 0 14px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    font-size: var(--fs-sm);
    color: var(--text-faint);
  }
  .dot {
    padding: 0 2px;
  }

  footer button {
    padding: 3px 9px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    background: var(--bg-panel);
    font: inherit;
    color: var(--text);
    cursor: pointer;
  }
  footer button:hover {
    background: var(--bg-active);
  }
</style>
