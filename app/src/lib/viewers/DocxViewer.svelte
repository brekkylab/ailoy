<script lang="ts">
  import type { BytesProps } from './registry';

  let { entry, bytes }: BytesProps = $props();

  let host = $state<HTMLDivElement | undefined>(undefined);
  let status = $state<'rendering' | 'ready' | 'failed'>('rendering');

  // Rendering writes into the DOM rather than returning markup, so it is an effect over
  // the element and not a `$derived` — and it renders into a detached element that is
  // swapped in whole once it is finished. That is what keeps a half-built page off the
  // screen, and what keeps a render that is still running when the viewer is pointed at
  // another file from writing its pages into the one now on screen.
  //
  // The stylesheet goes into the same detached element. A style element applies to the
  // whole document wherever it sits, so what makes this safe is that every rule it
  // writes is under `.docx`; what the placement buys is that the sheet is discarded
  // along with the element instead of accumulating, one per file opened.
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
        // Logged whether or not this render is still the current one, because what
        // goes on screen is deliberately not this: the text a failure carries is
        // written for whoever is reading the console, and names parts of a file
        // format that mean nothing to someone who opened a form.
        console.warn(`${entry.name} could not be rendered`, error);
        if (!live) return;
        status = 'failed';
      });

    return () => {
      live = false;
    };
  });

  async function render(source: ArrayBuffer, into: HTMLElement): Promise<void> {
    // Loaded when a document is opened rather than when the app starts. The renderer
    // is far larger than the app around it, and most sessions never open a `.docx`.
    const { renderAsync } = await import('docx-preview');
    await renderAsync(source, into, into, {
      // The document's page geometry is the point of a form: the margins and the
      // column widths are what make a filled-in cell land where it does on paper.
      inWrapper: true,
      breakPages: true,
      ignoreWidth: false,
      ignoreHeight: false,
      // The fonts a `.docx` embeds are not installed here, so the substitution is the
      // browser's either way. Honouring the names asked for is what gets the CJK face
      // these documents are written in.
      ignoreFonts: false,
      renderHeaders: true,
      renderFooters: true,
      renderFootnotes: true,
      renderEndnotes: true,
      // Tracked changes as the author left them: a form that went round for approval
      // reads wrong with the edits silently applied.
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
      <!-- The two ways a `.docx` that is not one gets here. Both are about the file
           rather than the viewer, which is what makes them worth naming. -->
      <p class="hint">
        A <code>.doc</code> saved under a <code>.docx</code> name, or a download that
        did not finish, both land here.
      </p>
    </div>
  {/if}
  <!-- Present from the first paint, so the effect above has somewhere to put the pages
       once they are built. Empty until then. -->
  <div class="page" bind:this={host}></div>
</div>

<style>
  .stage {
    min-height: 100%;
    /* The letterboxing around a page, as in the PDF viewer: what is inside is the
       document's own colouring, and it is not the panel's to restyle. */
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

  /* The rendered document is written into `.page` by the effect, so everything under
     it is global. The sheet the renderer brings styles the document itself; these
     rules only place the pages it produces. */
  .page :global(.docx-wrapper) {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    /* As in the spreadsheet viewer: sized to its content rather than to the panel, so
       that a page wider than the panel — a landscape document — keeps its padding and
       scrolls instead of being centred half out of view. */
    box-sizing: border-box;
    width: max-content;
    min-width: 100%;
    padding: 24px;
    background: transparent;
  }
  .page :global(.docx-wrapper > section.docx) {
    /* A page narrower than the panel keeps its width; a wider one scrolls rather than
       being squeezed, because a form's column widths are the layout. */
    box-shadow: var(--shadow-md);
    background: #ffffff;
  }
</style>
