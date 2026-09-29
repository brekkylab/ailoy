<script lang="ts">
  import { readSheet } from './sheet';
  import type { Sheet } from './sheet';
  import type { BytesProps } from './registry';

  let { entry, bytes }: BytesProps = $props();

  let sheets = $state<Sheet[]>([]);
  let active = $state(0);
  let status = $state<'reading' | 'ready' | 'failed'>('reading');

  $effect(() => {
    const source = bytes;
    let live = true;
    status = 'reading';
    sheets = [];
    active = 0;

    read(source)
      .then((read) => {
        if (!live) return;
        sheets = read;
        status = read.length === 0 ? 'failed' : 'ready';
      })
      .catch((error: unknown) => {
        // Reader errors name file-format parts, which do not belong in front of a form; they go
        // to the console and the panel shows the actionable message.
        console.warn(`${entry.name} could not be read`, error);
        if (live) status = 'failed';
      });

    return () => {
      live = false;
    };
  });

  async function read(source: ArrayBuffer): Promise<Sheet[]> {
    // Loaded on first open: the reader is far larger than the app, and most sessions never
    // open a spreadsheet.
    const { Workbook } = await import('exceljs');
    const workbook = new Workbook();
    // `repair` first: workbooks the agent writes wire their parts in a way the reader does
    // not resolve. Files that need no repair come back as the same bytes.
    await workbook.xlsx.load(await repair(source));
    return workbook.worksheets.map(readSheet);
  }

  let sheet = $derived(sheets[active]);
</script>

{#if status === 'reading'}
  <p class="note">Reading {entry.name}…</p>
{:else if status === 'failed'}
  <div class="note stack">
    <p>This file could not be read as a spreadsheet.</p>
    <!-- The two ways a non-`.xlsx` file with that extension arrives. Both are about the
         file, not the viewer, so they are named. -->
    <p class="hint">
      An <code>.xls</code> saved under an <code>.xlsx</code> name, or a download that
      did not finish, both land here.
    </p>
  </div>
{:else if sheet}
  <div class="stage">
    <div class="paper">
      <table style="width: {sheet.widths.reduce((a, b) => a + b, 0)}px">
        <colgroup>
          {#each sheet.widths as width}<col style="width: {width}px" />{/each}
        </colgroup>
        <tbody>
          {#each sheet.rows as row}
            <tr style="height: {row.height}px">
              {#each row.cells as cell}
                <td
                  colspan={cell.colSpan > 1 ? cell.colSpan : undefined}
                  rowspan={cell.rowSpan > 1 ? cell.rowSpan : undefined}
                  class:wrap={cell.style.wrap}
                  style:background={cell.style.fill}
                  style:color={cell.style.color}
                  style:font-weight={cell.style.bold ? '700' : null}
                  style:font-style={cell.style.italic ? 'italic' : null}
                  style:text-decoration={cell.style.underline ? 'underline' : null}
                  style:font-size={cell.style.size ? `${cell.style.size}px` : null}
                  style:font-family={cell.style.family}
                  style:text-align={cell.style.align}
                  style:vertical-align={cell.style.valign}
                  style:border-top={border(cell.style.borders.top)}
                  style:border-right={border(cell.style.borders.right)}
                  style:border-bottom={border(cell.style.borders.bottom)}
                  style:border-left={border(cell.style.borders.left)}>{cell.text}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <footer>
    {#if sheets.length > 1}
      <!-- The workbook's tabs in its own order; hidden for a single sheet (a form), where one
           tab is not a choice. -->
      <div class="tabs" role="tablist" aria-label="Sheets">
        {#each sheets as tab, i}
          <button
            role="tab"
            aria-selected={i === active}
            class:on={i === active}
            onclick={() => (active = i)}>{tab.name}</button>
        {/each}
      </div>
    {:else}
      <span class="name">{sheet.name}</span>
    {/if}
    <span class="count">
      {sheet.rows.length.toLocaleString()}
      {sheet.rows.length === 1 ? 'row' : 'rows'}
      <span class="dot">·</span>
      {sheet.widths.length}
      {sheet.widths.length === 1 ? 'column' : 'columns'}
      {#if sheet.truncated}<span class="dot">·</span>cut off here{/if}
    </span>
  </footer>
{/if}

<script lang="ts" module>
  import type { Border } from './sheet';

  /// A side as the CSS shorthand, or `null` to let the neighbouring cell's border draw the
  /// line.
  function border(side: Border | null): string | null {
    return side && `${side.width}px ${side.style} ${side.color}`;
  }

  // -- Making an openpyxl workbook readable by the reader -----------
  //
  // An `.xlsx` is a zip of XML parts wired by `.rels` files. Excel writes relationship
  // targets relative to the holding part (`../tables/table1.xml`), and `exceljs` indexes
  // and looks up parts by that exact string, with no path resolution. openpyxl (which the
  // agent uses) writes the same target absolutely (`/xl/tables/table1.xml`). Both are valid
  // OPC and Excel opens either, but `exceljs` finds nothing for the absolute form and the
  // load throws, so such a file fails here yet opens fine in Excel.
  //
  // So the bytes are rewritten first: absolute targets become the relative spelling
  // `exceljs` indexes by, and cell comments are dropped. openpyxl puts comments at
  // `xl/comments/comment1.xml` while `exceljs` only looks for `xl/comments1.xml`, so no
  // target spelling finds them, and the viewer does not render notes anyway.
  //
  // A workbook needing none of this is returned untouched, so Excel's own files skip
  // repacking.

  /// Relationship types whose parts are dropped rather than rewired, matched against the
  /// tail of the `Type` URI.
  const DROPPED = /\/(?:comments|vmlDrawing)$/;

  /// Where openpyxl puts those parts. Removed along with the relationships pointing at them,
  /// so nothing addresses a missing part.
  const DROPPED_PARTS = /^xl\/comments\/|\.vml$/i;

  /// One whole `<Relationship>` element: self-closing (as both writers emit) or an
  /// open/close pair (also schema-valid).
  const RELATIONSHIP = /<Relationship\b[^>]*?(?:\/>|>[\s\S]*?<\/Relationship>)/g;

  const ATTR = (name: string) => new RegExp(`\\b${name}="([^"]*)"`);

  /// An absolute part name as the relative path `exceljs` would index it under.
  ///
  /// Must be the shortest relative path, the one `exceljs` itself writes: shared leading
  /// folders dropped, one `../` per remaining folder. From `xl/worksheets/`,
  /// `xl/tables/table1.xml` is `../tables/table1.xml`, never `../../xl/tables/table1.xml`,
  /// which names the same file but would not be found.
  function relativize(fromDir: string, absolute: string): string {
    const from = fromDir.split('/').filter(Boolean);
    const to = absolute.split('/').filter(Boolean);
    let shared = 0;
    // `to.length - 1` excludes the filename: a part is never its own folder.
    while (shared < from.length && shared < to.length - 1 && from[shared] === to[shared]) shared++;
    return '../'.repeat(from.length - shared) + to.slice(shared).join('/');
  }

  /// One `.rels` part rewritten, or null if already in the reader's shape; all-null means
  /// the workbook needs no repacking.
  function rewrite(xml: string, dir: string): string | null {
    let changed = false;
    const out = xml.replace(RELATIONSHIP, (element) => {
      if (DROPPED.test(ATTR('Type').exec(element)?.[1] ?? '')) {
        changed = true;
        return '';
      }
      // A target outside the package is a URL, not a part name; left as is.
      if (ATTR('TargetMode').exec(element)?.[1] === 'External') return element;
      const target = ATTR('Target').exec(element)?.[1];
      if (!target || !target.startsWith('/')) return element;
      changed = true;
      // Function replacement, so a `$` in a part name is literal, not a backreference.
      return element.replace(ATTR('Target'), () => `Target="${relativize(dir, target.slice(1))}"`);
    });
    return changed ? out : null;
  }

  /// `bytes` in a shape `exceljs` can load; the same bytes if already so.
  ///
  /// Never throws: a non-zip or unrelated zip is returned unchanged, so the viewer reports
  /// the reader's own error rather than this pass's.
  async function repair(bytes: ArrayBuffer): Promise<ArrayBuffer> {
    try {
      // Loaded with the reader, on first spreadsheet open. `exceljs` unpacks zips with this
      // same library, so it adds no cost.
      const JSZip = (await import('jszip')).default;
      const zip = await JSZip.loadAsync(bytes);

      let changed = false;
      for (const path of Object.keys(zip.files)) {
        if (!/_rels\/[^/]*\.rels$/.test(path)) continue;
        const dir = path.replace(/_rels\/[^/]*$/, '');
        const rewritten = rewrite(await zip.files[path].async('string'), dir);
        if (rewritten === null) continue;
        zip.file(path, rewritten);
        changed = true;
      }
      if (!changed) return bytes;

      for (const path of Object.keys(zip.files)) {
        if (DROPPED_PARTS.test(path)) zip.remove(path);
      }
      // Stored, not deflated: read once in memory right away, so compressing is wasted work.
      return await zip.generateAsync({ type: 'arraybuffer' });
    } catch {
      return bytes;
    }
  }
</script>

<style>
  .stage {
    /* Letterboxing around the sheet: inside is the document's own colouring, not the panel's
       to restyle. `max-content`, not a plain block, because a wider-than-panel sheet
       scrolls: a block is only as wide as its container, so its right padding would land
       under the sheet and the last column would sit flush against the edge. */
    box-sizing: border-box;
    width: max-content;
    min-width: 100%;
    min-height: 100%;
    padding: 24px;
    background: var(--bg-sunken);
  }

  .paper {
    width: max-content;
    margin: 0 auto;
    background: #ffffff;
    box-shadow: var(--shadow-md);
  }

  table {
    border-collapse: collapse;
    /* Fixed, so the file's column widths are the layout rather than a starting point the
       browser rebalances (which in a form puts values under the wrong heading). */
    table-layout: fixed;
    color: #1a1917;
    font-size: var(--fs-md);
  }

  td {
    padding: 2px 5px;
    overflow: hidden;
    /* Cells do not wrap unless the cell says so; form column widths assume that. */
    white-space: pre;
    text-overflow: ellipsis;
    vertical-align: bottom;
  }
  td.wrap { white-space: pre-wrap; overflow-wrap: anywhere; }

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

  footer {
    position: sticky;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 7px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    font-size: var(--fs-sm);
    color: var(--text-faint);
  }
  .name { font-weight: 600; color: var(--text-muted); }
  .count { margin-left: auto; }
  .dot { padding: 0 2px; }

  .tabs { display: flex; gap: 2px; overflow-x: auto; }
  .tabs button {
    flex: none;
    padding: 3px 9px;
    border: none;
    border-radius: var(--radius);
    background: none;
    color: var(--text-muted);
    font: inherit;
    cursor: pointer;
  }
  .tabs button:hover { background: var(--bg-hover); color: var(--text); }
  .tabs button.on { background: var(--bg-active); color: var(--text); font-weight: 600; }
</style>
