<script lang="ts">
  import { File, X } from '@lucide/svelte';

  import { btn } from '../ui';
  import type { Entry } from './entry';
  import type { Decoded } from '../encoding';
  import { formatSize } from '../files';
  import type { Source } from './source';
  import { viewerFor } from './registry';

  // Reads and renders one file, without chrome. `unsupported` and `too-big` are not read
  // failures: the file is fine, and each says why it is not on screen.

  interface Props {
    entry: Entry;
    /// The tree this file is in (see `source.ts`).
    source: Source;
  }

  let { entry, source }: Props = $props();

  let viewer = $derived(viewerFor(entry));
  let status = $state<'loading' | 'ready' | 'failed' | 'too-big' | 'unsupported'>('loading');
  let text = $state('');
  /// The encoding `text` was decoded from, passed on for the viewer to show. UTF-8 until a
  /// read says otherwise.
  let encoding = $state<Decoded['encoding']>('UTF-8');
  let bytes = $state<ArrayBuffer | null>(null);

  $effect(() => {
    // Re-reads on another file, or the same one with a new size or date (e.g. the agent
    // rewrote it mid-conversation).
    void [entry.name, entry.segments.join('/'), entry.size, entry.modified];
    load();
  });

  async function load() {
    const target = entry;
    text = '';
    bytes = null;
    if (!viewer) {
      status = 'unsupported';
      return;
    }
    if (viewer.source === 'url') {
      // Nothing to fetch: the viewer gets the URL and the browser reads it.
      status = 'ready';
      return;
    }
    if (target.size != null && target.size > viewer.maxBytes) {
      status = 'too-big';
      return;
    }
    status = 'loading';
    try {
      if (viewer.source === 'bytes') {
        const body = await source.bytes(target);
        // The file may have been closed, or another opened, while this was in flight.
        if (target !== entry) return;
        bytes = body;
      } else {
        const body = await source.decoded(target);
        if (target !== entry) return;
        text = body.text;
        encoding = body.encoding;
      }
      status = 'ready';
    } catch {
      if (target !== entry) return;
      status = 'failed';
    }
  }
</script>

{#if status === 'loading'}
  <p class="note">Loading…</p>
{:else if status === 'ready' && viewer}
  {#if viewer.source === 'text'}
    {@const TextViewer = viewer.component}
    <!-- Relative links resolve against this file's own folder in its tree (see `source.ts`). -->
    <TextViewer {entry} {text} {encoding} base={source.base(entry)} />
  {:else if viewer.source === 'bytes'}
    {@const BytesViewer = viewer.component}
    <!-- Keyed on the file, so another file builds a fresh viewer rather than feeding new bytes
         to one that has already rendered. -->
    {#key entry}
      {#if bytes}<BytesViewer {entry} {bytes} />{/if}
    {/key}
  {:else}
    {@const UrlViewer = viewer.component}
    <UrlViewer {entry} url={source.url(entry)} />
  {/if}
{:else if status === 'unsupported'}
  <div class="note stack">
    <File class="size-6.5" strokeWidth={1.3} />
    <p>No viewer for this file type yet.</p>
  </div>
{:else if status === 'too-big'}
  <div class="note stack">
    <File class="size-6.5" strokeWidth={1.3} />
    <p>{formatSize(entry.size)} is too large to open here.</p>
  </div>
{:else}
  <div class="note stack">
    <X class="size-6.5" strokeWidth={1.3} />
    <p>That file could not be read.</p>
    <button class={btn.outline} onclick={load}>Retry</button>
  </div>
{/if}

<style>
  .note {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    font-size: var(--fs-md);
    line-height: 1.55;
    text-align: center;
    color: var(--text-faint);
  }
  .note.stack { flex-direction: column; gap: 10px; }
  .note p { margin: 0; }
</style>
