<script lang="ts">
  import type { TextProps } from './registry';

  let { text, encoding }: TextProps = $props();

  /// Lines per page in the DOM. Nightly logs run to tens of thousands of lines, and laying
  /// them all out at once hangs the tab.
  const PAGE = 2000;

  /// The file's lines with CRLF carriage returns stripped; left in, each lands at a line end
  /// and gets dragged along on copy. The footer notes when they were present.
  let crlf = $derived(text.includes('\r\n'));
  let lines = $derived(split(text));

  let shown = $state(PAGE);
  /// Off by default: `.txt` files here are mostly fixed-width reports or logs, whose columns
  /// wrapping breaks. The toggle is for prose.
  let wrap = $state(false);

  // Back to the first page on another file: the component is reused, not rebuilt.
  $effect(() => {
    void text;
    shown = PAGE;
  });

  let visible = $derived(lines.slice(0, shown));
  let remaining = $derived(lines.length - visible.length);

  /// `text` as lines. A trailing newline ends the last line rather than starting an empty
  /// one, matching editors and the line count.
  function split(source: string): string[] {
    const body = source.endsWith('\n') ? source.slice(0, -1) : source;
    if (body === '') return [];
    return body.split('\n').map((line) => (line.endsWith('\r') ? line.slice(0, -1) : line));
  }
</script>

{#if lines.length === 0}
  <p class="empty">This file is empty.</p>
{:else}
  <div class="sheet" class:wrap>
    <table>
      <tbody>
        {#each visible as line, i}
          <tr>
            <!-- The file's own line number, so a reader can cite a line; logs are quoted by line. -->
            <th class="gutter" scope="row">{i + 1}</th>
            <td class="line">{line}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <footer>
    <span>
      {lines.length.toLocaleString()}
      {lines.length === 1 ? 'line' : 'lines'}
      <span class="dot">·</span>
      {crlf ? 'CRLF' : 'LF'}
      <span class="dot">·</span>
      {encoding}
    </span>
    <span class="actions">
      {#if remaining > 0}
        <button onclick={() => (shown += PAGE)}>
          Show {Math.min(PAGE, remaining).toLocaleString()} more
          <span class="faint">({remaining.toLocaleString()} left)</span>
        </button>
      {/if}
      <button onclick={() => (wrap = !wrap)} aria-pressed={wrap}>
        {wrap ? 'No wrap' : 'Wrap'}
      </button>
    </span>
  </footer>
{/if}

<style>
  .sheet {
    /* Scrolls sideways for long lines while the panel behind scrolls down; the footer sits
       below. */
    min-height: calc(100% - 30px);
    overflow-x: auto;
  }
  /* When wrapped nothing overflows, so the table fills the panel rather than sizing to its
     widest line. */
  .sheet.wrap { overflow-x: hidden; }
  .sheet.wrap table { width: 100%; }
  .sheet.wrap .line { white-space: pre-wrap; overflow-wrap: anywhere; }

  table {
    border-collapse: separate;
    border-spacing: 0;
    font-family: var(--font-mono);
    font-size: var(--fs-md);
    line-height: 1.6;
  }

  .line {
    padding: 0 14px 0 12px;
    color: var(--text);
    /* Spacing is the layout of a report or log, so it is kept. */
    white-space: pre;
  }

  /* Line-number gutter, fixed during sideways scroll so each number stays with its line. */
  .gutter {
    position: sticky;
    left: 0;
    z-index: 1;
    padding: 0 10px;
    border-right: 1px solid var(--border);
    background: var(--bg-sunken);
    font-size: var(--fs-sm);
    font-weight: 400;
    color: var(--text-faint);
    text-align: right;
    vertical-align: top;
    user-select: none;
  }

  tr:hover .line,
  tr:hover .gutter {
    background-color: var(--bg-hover);
  }

  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    margin: 0;
    font-size: var(--fs-md);
    color: var(--text-faint);
  }

  footer {
    position: sticky;
    bottom: 0;
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
  .dot { padding: 0 2px; }
  .actions { display: flex; align-items: center; gap: 8px; }

  footer button {
    padding: 3px 9px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    background: var(--bg-panel);
    font: inherit;
    color: var(--text);
    cursor: pointer;
  }
  footer button:hover { background: var(--bg-active); }
  footer button[aria-pressed='true'] { background: var(--bg-active); }
  .faint { color: var(--text-faint); }
</style>
