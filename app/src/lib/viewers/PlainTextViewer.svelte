<script lang="ts">
  import type { TextProps } from './registry';

  let { text, encoding }: TextProps = $props();

  /// How many lines are put in the DOM at a time. A nightly log runs to tens of
  /// thousands of them, and a browser asked to lay all of that out at once is a tab
  /// that stops answering — so the first page is on screen immediately and the rest
  /// is a click away, as in the CSV viewer.
  const PAGE = 2000;

  /// The file's lines, with the carriage returns of a CRLF file taken off. Left on,
  /// each one is a character in the DOM that lands at the end of a line and takes the
  /// cursor with it on copy. The footer says when they were there.
  let crlf = $derived(text.includes('\r\n'));
  let lines = $derived(split(text));

  let shown = $state(PAGE);
  /// Off by default: what arrives as `.txt` here is a fixed-width report or a log,
  /// and wrapping breaks the columns it was aligned into. Prose is the case the
  /// toggle is for.
  let wrap = $state(false);

  // Back to the first page when the viewer is pointed at another file, which reuses
  // this component rather than building a new one.
  $effect(() => {
    void text;
    shown = PAGE;
  });

  let visible = $derived(lines.slice(0, shown));
  let remaining = $derived(lines.length - visible.length);

  /// `text` as lines. A file that ends in a newline ends a line rather than starting
  /// an empty one, which is what every editor shows and what the count should say.
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
            <!-- The number is the line's own, so a reader can point at a line in the
                 source — a log is quoted by line as often as it is read. -->
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
    /* The footer sits below; this is the part that scrolls sideways when a long line
       needs it, while the panel behind it scrolls down. */
    min-height: calc(100% - 30px);
    overflow-x: auto;
  }
  /* Wrapped, nothing overflows sideways, so the table fills the panel instead of
     sizing itself to its widest line. */
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
    /* The file's own spacing is the layout of a report or a log, so it is kept. */
    white-space: pre;
  }

  /* The line-number gutter: fixed while the text scrolls sideways, so the number
     stays with its line. */
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
