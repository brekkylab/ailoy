<script lang="ts">
  import { parseCsv, isNumericColumn } from './csv';
  import { extOf } from '../files';
  import type { TextProps } from './registry';

  let { entry, text, encoding }: TextProps = $props();

  /// How many records are put in the DOM at a time. A spreadsheet export runs to tens
  /// of thousands of rows, and a table that large lays out slowly enough to lock the
  /// tab — so the first page is on screen immediately and the rest is a click away.
  const PAGE = 500;

  // A `.tsv` has already said what its delimiter is; anything else is sniffed.
  let table = $derived(parseCsv(text, extOf(entry.name) === 'tsv' ? '\t' : undefined));
  let numeric = $derived(
    Array.from({ length: table.columns }, (_, i) => isNumericColumn(table.rows, i)),
  );

  let shown = $state(PAGE);
  // Back to the first page when the viewer is pointed at another file, which reuses
  // this component rather than building a new one.
  $effect(() => {
    void text;
    shown = PAGE;
  });

  let visible = $derived(table.rows.slice(0, shown));
  let remaining = $derived(table.rows.length - visible.length);

  const DELIMITER_NAMES: Record<string, string> = {
    ',': 'comma',
    ';': 'semicolon',
    '\t': 'tab',
    '|': 'pipe',
  };
</script>

{#if table.columns === 0}
  <p class="empty">This file has no rows.</p>
{:else}
  <div class="sheet">
    <table>
      <thead>
        <tr>
          <!-- The row numbers are the file's, not the table's, so a reader can point
               at a row in the source. The header record is row 1. -->
          <th class="gutter" scope="col"><span class="sr-only">Row</span></th>
          {#each table.header as cell, i}
            <th scope="col" class:num={numeric[i]}>{cell}</th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each visible as row, r}
          <tr>
            <th class="gutter" scope="row">{r + 2}</th>
            {#each row as cell, i}
              <td class:num={numeric[i]} class:blank={cell === ''}>{cell}</td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <footer>
    <span>
      {table.rows.length.toLocaleString()}
      {table.rows.length === 1 ? 'row' : 'rows'}
      <span class="dot">·</span>
      {table.columns}
      {table.columns === 1 ? 'column' : 'columns'}
      <span class="dot">·</span>
      {DELIMITER_NAMES[table.delimiter] ?? 'delimiter'}-separated
      <span class="dot">·</span>
      {encoding}
    </span>
    {#if remaining > 0}
      <button onclick={() => (shown += PAGE)}>
        Show {Math.min(PAGE, remaining).toLocaleString()} more
        <span class="faint">({remaining.toLocaleString()} left)</span>
      </button>
    {/if}
  </footer>
{/if}

<style>
  .sheet {
    /* The footer sits below; this is the part that scrolls sideways when a wide sheet
       needs it, while the panel behind it scrolls down. */
    min-height: calc(100% - 30px);
    overflow-x: auto;
  }

  table {
    border-collapse: separate;
    border-spacing: 0;
    font-size: var(--fs-md);
    line-height: 1.5;
    white-space: pre;
  }

  th,
  td {
    padding: 5px 12px;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
    /* A single cell holding a paragraph should not push every column off screen. */
    max-width: 42ch;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--bg-sunken);
    font-weight: 600;
    color: var(--text);
  }

  /* The row-number gutter: fixed while the sheet scrolls sideways, so the number
     stays with its row. */
  .gutter {
    position: sticky;
    left: 0;
    z-index: 1;
    padding: 5px 10px;
    background: var(--bg-sunken);
    font-family: var(--font-mono);
    font-size: var(--fs-sm);
    font-weight: 400;
    color: var(--text-faint);
    text-align: right;
    user-select: none;
  }
  thead .gutter {
    z-index: 2;
  }

  tbody tr:hover td,
  tbody tr:hover .gutter {
    /* `background-color`, not the shorthand: the shorthand would clear the blank-cell
       marker below, which is a background-image. */
    background-color: var(--bg-hover);
  }

  .num {
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .blank {
    /* An empty cell reads as an empty cell rather than as a rendering slip. */
    background-image: linear-gradient(
      to top right,
      transparent calc(50% - 0.5px),
      var(--border) calc(50% - 0.5px),
      var(--border) calc(50% + 0.5px),
      transparent calc(50% + 0.5px)
    );
    background-size: 7px 7px;
    background-repeat: no-repeat;
    background-position: 12px center;
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
  .faint {
    color: var(--text-faint);
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

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
</style>
