<script lang="ts">
  import { File, X } from '@lucide/svelte';

  import { btn } from '../ui';
  import type { Entry } from './entry';
  import type { Decoded } from '../encoding';
  import { formatSize } from '../files';
  import type { Source } from './source';
  import { viewerFor } from './registry';

  // One file, read and rendered — and nothing around it.
  //
  // This is the half of opening a file that is the same wherever it is opened: work out
  // which viewer the name calls for, read the file through whatever tree it is in, and
  // hand the viewer what its kind takes. The chrome around it is the caller's.
  //
  // `unsupported` and `too-big` are answers about the file and not failures to read it:
  // the file is fine, and each says why it is not on screen.

  interface Props {
    entry: Entry;
    /// Which tree this file is in. See `source.ts`.
    source: Source;
  }

  let { entry, source }: Props = $props();

  let viewer = $derived(viewerFor(entry));
  let status = $state<'loading' | 'ready' | 'failed' | 'too-big' | 'unsupported'>('loading');
  let text = $state('');
  /// What `text` was decoded from — handed on to the viewer, which is where there is
  /// room to say so. UTF-8 until a read says otherwise.
  let encoding = $state<Decoded['encoding']>('UTF-8');
  let bytes = $state<ArrayBuffer | null>(null);

  $effect(() => {
    // Re-reads when pointed at another file, and when the same one comes back with a
    // new size or date — a file the agent rewrote mid-conversation is a new read.
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
      // Nothing to fetch: the viewer is handed the address and the browser reads it.
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
        // The file may have been closed, or another one opened, while this was in
        // flight.
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
    <!-- The base a relative link resolves against is this file's own folder in its own
         tree — see `source.ts`. -->
    <TextViewer {entry} {text} {encoding} base={source.base(entry)} />
  {:else if viewer.source === 'bytes'}
    {@const BytesViewer = viewer.component}
    <!-- Keyed on the file, so opening another builds a fresh viewer rather than handing
         new bytes to one that has already rendered. -->
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
