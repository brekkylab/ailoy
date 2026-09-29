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
        // What a reader says about a malformed file names parts of a file format,
        // which is not what belongs on screen in front of a form. It goes to the
        // console, and the panel says the one thing that is actionable.
        console.warn(`${entry.name} could not be read`, error);
        if (live) status = 'failed';
      });

    return () => {
      live = false;
    };
  });

  async function read(source: ArrayBuffer): Promise<Sheet[]> {
    // Loaded when a spreadsheet is opened rather than when the app starts. The reader
    // is far larger than the app around it, and most sessions never open one.
    const { Workbook } = await import('exceljs');
    const workbook = new Workbook();
    // Not `source` directly: a workbook the agent wrote wires its parts up in a way the
    // reader does not resolve, and `repair` is what puts it in the shape it reads. A
    // file that does not need it comes back as the same bytes.
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
    <!-- The two ways an `.xlsx` that is not one gets here. Both are about the file
         rather than the viewer, which is what makes them worth naming. -->
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
      <!-- The workbook's own tabs, in its own order. Hidden for the single-sheet case,
           which is what a form is, because one tab is not a choice. -->
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

  /// A side, as the CSS shorthand — or `null`, which leaves the neighbouring cell's
  /// border to draw the line.
  function border(side: Border | null): string | null {
    return side && `${side.width}px ${side.style} ${side.color}`;
  }

  // -- Making an openpyxl workbook readable by the reader -----------
  //
  // An `.xlsx` is a zip of XML parts wired together by `.rels` files, and the wiring is
  // the one place the two ends of this app disagree. Excel writes a relationship target
  // relative to the part that holds it — a sheet points at its table as
  // `../tables/table1.xml` — and `exceljs` takes that string at face value: it indexes
  // the parts it unpacked under exactly that spelling and looks them up by it, with no
  // path resolution in between. openpyxl, which is what the agent writes workbooks with,
  // spells the same target absolutely: `/xl/tables/table1.xml`. Both are valid OPC, Excel
  // opens either, and `exceljs` finds nothing under the second one — so the lookup
  // returns `undefined` and the load throws on the next line. That is why a file the
  // browser could not open downloads and opens fine in Excel.
  //
  // So the bytes are rewritten before the reader sees them: absolute targets become the
  // relative spelling `exceljs` indexes by, and cell comments are dropped. Comments go
  // rather than get rewritten because openpyxl puts them at `xl/comments/comment1.xml`
  // while `exceljs` only ever looks for `xl/comments1.xml` — there is no target spelling
  // that would find them — and the viewer does not render notes anyway, so nothing that
  // reaches the screen is lost.
  //
  // A workbook that needs none of this is handed back untouched, bytes and all: Excel's
  // own files take the reader's fast path and never get repacked.

  /// A relationship type whose part this viewer drops rather than rewires. Matched
  /// against the tail of the `Type` URI.
  const DROPPED = /\/(?:comments|vmlDrawing)$/;

  /// Where those parts live in a workbook openpyxl wrote. Removed alongside the
  /// relationships that point at them, so nothing is left addressing a part that is gone.
  const DROPPED_PARTS = /^xl\/comments\/|\.vml$/i;

  /// One `<Relationship>` element, whole — self-closing, which is how both writers spell
  /// it, or an open/close pair, which the schema also allows.
  const RELATIONSHIP = /<Relationship\b[^>]*?(?:\/>|>[\s\S]*?<\/Relationship>)/g;

  const ATTR = (name: string) => new RegExp(`\\b${name}="([^"]*)"`);

  /// An absolute part name, as the relative path `exceljs` would have indexed it under.
  ///
  /// Not any relative path that resolves to the same part — the one it writes itself,
  /// which is the shortest: the shared leading folders are dropped and one `../` is
  /// spent for each folder left over on the way up. `xl/worksheets/` to
  /// `xl/tables/table1.xml` is `../tables/table1.xml`, and never
  /// `../../xl/tables/table1.xml`, which addresses the same file and would still not be
  /// found.
  function relativize(fromDir: string, absolute: string): string {
    const from = fromDir.split('/').filter(Boolean);
    const to = absolute.split('/').filter(Boolean);
    let shared = 0;
    // `to.length - 1` keeps the filename out of it: a part is never its own folder.
    while (shared < from.length && shared < to.length - 1 && from[shared] === to[shared]) shared++;
    return '../'.repeat(from.length - shared) + to.slice(shared).join('/');
  }

  /// One `.rels` part, rewritten — or null when it was already in the shape the reader
  /// wants, which is what says a workbook needs no repacking at all.
  function rewrite(xml: string, dir: string): string | null {
    let changed = false;
    const out = xml.replace(RELATIONSHIP, (element) => {
      if (DROPPED.test(ATTR('Type').exec(element)?.[1] ?? '')) {
        changed = true;
        return '';
      }
      // A target outside the package is a URL, not a part name, and is left as it stands.
      if (ATTR('TargetMode').exec(element)?.[1] === 'External') return element;
      const target = ATTR('Target').exec(element)?.[1];
      if (!target || !target.startsWith('/')) return element;
      changed = true;
      // A function replacement, so a `$` in a part's name is a `$` and not a backreference.
      return element.replace(ATTR('Target'), () => `Target="${relativize(dir, target.slice(1))}"`);
    });
    return changed ? out : null;
  }

  /// `bytes` in a shape `exceljs` can load — the same bytes when they already were.
  ///
  /// Never throws on a file it cannot make sense of: something that is not a zip, or is
  /// a zip of something else, is handed back unchanged so that the error the viewer
  /// reports is the reader's own rather than this pass failing first.
  async function repair(bytes: ArrayBuffer): Promise<ArrayBuffer> {
    try {
      // Alongside the reader itself: both are loaded when a spreadsheet is opened and
      // not before. `exceljs` unpacks zips with this same library, so it costs nothing
      // that was not already being paid.
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
      // Stored rather than deflated: this archive is read once, in memory, by the line
      // that asked for it, and compressing it would be work done to be undone.
      return await zip.generateAsync({ type: 'arraybuffer' });
    } catch {
      return bytes;
    }
  }
</script>

<style>
  .stage {
    /* The letterboxing around the sheet, as in the PDF and Word viewers: what is
       inside is the document's own colouring and not the panel's to restyle.
       `max-content` rather than a plain block, because a sheet wider than the panel
       scrolls: a block is only as wide as the container, so its right-hand padding
       lands under the sheet and the last column ends up flush against the edge. */
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
    /* Fixed, so the file's column widths are the layout rather than a starting point
       the browser rebalances against the content — which in a form is what puts a
       value under the wrong heading. */
    table-layout: fixed;
    color: #1a1917;
    font-size: var(--fs-md);
  }

  td {
    padding: 2px 5px;
    overflow: hidden;
    /* A spreadsheet does not wrap unless the cell says to, and a form's column widths
       are set on the assumption that it does not. */
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
