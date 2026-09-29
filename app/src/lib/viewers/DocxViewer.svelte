<script lang="ts">
  import type { BytesProps } from './registry';

  let { entry, bytes }: BytesProps = $props();

  let host = $state<HTMLDivElement | undefined>(undefined);
  let status = $state<'rendering' | 'ready' | 'failed'>('rendering');

  // The renderer writes into the DOM, so this is an effect, not a `$derived`. It renders
  // into a detached element swapped in whole when done, which keeps half-built pages off
  // screen and stops a stale render (after switching files) from writing into the current one.
  //
  // The stylesheet goes into the same detached element. Style elements apply document-wide
  // wherever they sit, so safety comes from every rule being under `.docx`; the placement
  // just discards the sheet with the element instead of accumulating one per file.
  $effect(() => {
    const target = host;
    const source = bytes;
    if (!target) return;

    let live = true;
    status = 'rendering';

    const staging = document.createElement('div');

    render(source, staging)
      .then(() => {
        if (!live) return;
        target.replaceChildren(...staging.childNodes);
        status = 'ready';
      })
      .catch((error: unknown) => {
        // Logged even if this render is stale. The error text is for the console: it names
        // file-format parts that mean nothing to someone who opened a form.
        console.warn(`${entry.name} could not be rendered`, error);
        if (!live) return;
        status = 'failed';
      });

    return () => {
      live = false;
    };
  });

  async function render(source: ArrayBuffer, into: HTMLElement): Promise<void> {
    // Loaded on first open: the renderer is far larger than the app, and most sessions never
    // open a `.docx`.
    const { renderAsync } = await import('docx-preview');
    await renderAsync(source, into, into, {
      // Page geometry is the point of a form: margins and column widths place each filled-in
      // cell where it lands on paper.
      inWrapper: true,
      breakPages: true,
      ignoreWidth: false,
      ignoreHeight: false,
      // Embedded fonts are not installed, so the browser substitutes either way; honouring the
      // requested names gets the CJK face these documents use.
      ignoreFonts: false,
      renderHeaders: true,
      renderFooters: true,
      renderFootnotes: true,
      renderEndnotes: true,
      // Tracked changes shown as left: an approval-routed form reads wrong with edits silently
      // applied.
      renderChanges: true,
      trimXmlDeclaration: true,
    });
  }
</script>

<div class="stage" class:busy={status === 'rendering'}>
  {#if status === 'rendering'}
    <p class="note">Rendering {entry.name}…</p>
  {:else if status === 'failed'}
    <div class="note stack">
      <p>This file could not be read as a Word document.</p>
      <!-- The two ways a non-`.docx` file with that extension arrives. Both are about the
           file, not the viewer, so they are named. -->
      <p class="hint">
        A <code>.doc</code> saved under a <code>.docx</code> name, or a download that
        did not finish, both land here.
      </p>
    </div>
  {/if}
  <!-- Present from first paint so the effect has a target for the built pages; empty until
       then. -->
  <div class="page" bind:this={host}></div>
</div>

<style>
  .stage {
    min-height: 100%;
    /* Letterboxing around the page: inside is the document's own colouring, not the panel's
       to restyle. */
    background: var(--bg-sunken);
  }
  .stage.busy { display: grid; place-items: center; }

  .note {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    font-size: var(--fs-md);
    color: var(--text-faint);
  }
  .note.stack { flex-direction: column; gap: 6px; }
  .note p { margin: 0; }
  .hint {
    max-width: 44ch;
    text-align: center;
    text-wrap: balance;
  }
  .hint code {
    padding: 0.1em 0.3em;
    border-radius: 4px;
    background: var(--bg-active);
    font-family: var(--font-mono);
    font-size: 0.9em;
  }

  /* The effect writes the rendered document into `.page`, so these rules are global. The
     renderer's own sheet styles the document; these only place its pages. */
  .page :global(.docx-wrapper) {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    /* Sized to content, not the panel, so a page wider than the panel (landscape) keeps its
       padding and scrolls instead of being centred half out of view. */
    box-sizing: border-box;
    width: max-content;
    min-width: 100%;
    padding: 24px;
    background: transparent;
  }
  .page :global(.docx-wrapper > section.docx) {
    /* Wider pages scroll rather than squeeze, because a form's column widths are the layout. */
    box-shadow: var(--shadow-md);
    background: #ffffff;
  }
</style>
