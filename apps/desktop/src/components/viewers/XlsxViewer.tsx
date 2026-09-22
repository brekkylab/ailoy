// A workbook, one sheet at a time, drawn the way the file says to.
//
// `lib/sheet` flattens a worksheet into a grid — the file's own column widths and row
// heights, its merges, fills, borders, fonts and number formats — and this puts that grid
// on paper. A spreadsheet is a laid-out document as much as a Word file is: a form whose
// columns were sized around its labels reads wrong at any other width, so the widths are
// the layout and the table does not rebalance them against its content.
//
// White paper on a sunken stage, as the Word viewer has: what is inside is the file's own
// colouring, and a pane that tinted it would be arguing with the document. The sheet tabs
// are the workbook's own, in its own order.
//
// `exceljs` is imported lazily — it is among the largest dependencies here and only a
// workbook needs it.

import { useEffect, useState } from "react";

import { cn } from "cn";

import { report } from "@/lib/report";
import type { Border, Cell, Sheet } from "@/lib/sheet";
import { useBytes } from "@/lib/useBytes";
import { S } from "@/strings";

/** A worksheet's pointer at its drawing. Empty by schema, but closed either way. */
const DRAWING_REF = /<drawing\b[^>]*\/>|<drawing\b[^>]*>[\s\S]*?<\/drawing>/g;

/**
 * The same workbook with every drawing removed — the parts and the worksheets' pointers
 * at them.
 *
 * For a workbook `exceljs` will not open. It matches a drawing's root element as the
 * literal tag `xdr:wsDr`, so a file that declares that namespace as the default and writes
 * `<wsDr>` — valid XML, and what more than one generator emits — parses to nothing, and
 * then two separate places read `.anchors` off the nothing without checking. One is
 * reached by the worksheet's `<drawing/>` element alone, which is why the parts have to go
 * *and* the pointers.
 *
 * Nothing is lost here: a drawing is a chart or an image, and this viewer has only ever
 * shown cells. It is done on the failure and not on every workbook, because rewriting the
 * archive costs a pass over the whole of it.
 */
async function withoutDrawings(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  const JSZip = (await import("jszip")).default;
  const zip = await JSZip.loadAsync(bytes);
  for (const name of Object.keys(zip.files)) {
    if (name.startsWith("xl/drawings/")) zip.remove(name);
  }
  for (const name of Object.keys(zip.files)) {
    if (!/^xl\/worksheets\/[^/]+\.xml$/.test(name)) continue;
    const xml = await zip.files[name].async("string");
    zip.file(name, xml.replace(DRAWING_REF, ""));
  }
  return zip.generateAsync({ type: "arraybuffer" });
}

async function readWorkbook(bytes: ArrayBuffer): Promise<Sheet[]> {
  const [{ Workbook }, { readSheet }] = await Promise.all([
    import("exceljs"),
    import("@/lib/sheet"),
  ]);
  let workbook = new Workbook();
  try {
    await workbook.xlsx.load(bytes);
  } catch (err) {
    report("a drawing exceljs cannot read; opening the workbook without it", err);
    // A fresh workbook rather than the one that threw: it stopped part-way through
    // loading, and what it holds is whatever it had got to.
    workbook = new Workbook();
    await workbook.xlsx.load(await withoutDrawings(bytes));
  }
  return workbook.worksheets.map(readSheet);
}

/** A side of a cell, as the CSS shorthand — or undefined, leaving the neighbour to draw it. */
function border(side: Border | null): string | undefined {
  return side ? `${side.width}px ${side.style} ${side.color}` : undefined;
}

function cellStyle(cell: Cell): React.CSSProperties {
  const s = cell.style;
  return {
    background: s.fill ?? undefined,
    color: s.color ?? undefined,
    fontWeight: s.bold ? 700 : undefined,
    fontStyle: s.italic ? "italic" : undefined,
    textDecoration: s.underline ? "underline" : undefined,
    fontSize: s.size ? `${s.size}px` : undefined,
    fontFamily: s.family ?? undefined,
    textAlign: s.align ?? undefined,
    verticalAlign: s.valign ?? undefined,
    borderTop: border(s.borders.top),
    borderRight: border(s.borders.right),
    borderBottom: border(s.borders.bottom),
    borderLeft: border(s.borders.left),
  };
}

export function XlsxViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const [sheets, setSheets] = useState<Sheet[] | null>(null);
  const [active, setActive] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (bytes.state !== "ready") return;
    let live = true;
    // No reset here: `FileBrowser` keys this on the path, so another workbook is another
    // mount and the state above starts where it should.
    void readWorkbook(bytes.bytes)
      .then((read) => {
        if (live) setSheets(read);
      })
      .catch((err: unknown) => {
        report(`${path} could not be read as a workbook`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="p-4 text-xs text-destructive">{S.viewerFailed}</p>;
  if (!sheets) return <p className="p-4 text-xs text-muted-foreground">{S.loading}</p>;
  if (sheets.length === 0) return <p className="p-4 text-xs text-muted-foreground">{S.empty}</p>;

  const sheet = sheets[Math.min(active, sheets.length - 1)];
  const width = sheet.widths.reduce((a, b) => a + b, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-auto bg-muted">
        {/* `w-max min-w-full` rather than a plain block: a sheet wider than the pane
            scrolls, and a block is only as wide as its container — so its right-hand
            padding would land *under* the sheet and the last column end up flush. */}
        <div className="w-max min-w-full p-6">
          <div className="mx-auto w-max bg-white shadow-md">
            {/* Fixed, so the file's widths are the layout rather than a starting point the
                browser rebalances against the content — which in a form is what puts a
                value under the wrong heading. */}
            <table className="table-fixed border-collapse text-xs text-black" style={{ width }}>
              <colgroup>
                {sheet.widths.map((w, i) => (
                  <col key={i} style={{ width: w }} />
                ))}
              </colgroup>
              <tbody>
                {sheet.rows.map((row, r) => (
                  <tr key={r} style={{ height: row.height }}>
                    {row.cells.map((cell, c) => (
                      <td
                        key={c}
                        colSpan={cell.colSpan > 1 ? cell.colSpan : undefined}
                        rowSpan={cell.rowSpan > 1 ? cell.rowSpan : undefined}
                        // A spreadsheet does not wrap unless the cell says to, and a form's
                        // column widths are set on the assumption that it does not.
                        className={cn(
                          "overflow-hidden px-[5px] py-[2px] align-bottom text-ellipsis",
                          cell.style.wrap ? "break-words whitespace-pre-wrap" : "whitespace-pre",
                        )}
                        style={cellStyle(cell)}
                      >
                        {cell.text}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="flex shrink-0 items-center justify-between gap-3 border-t px-3 py-1.5 text-xs text-muted-foreground">
        {sheets.length > 1 ? (
          // The workbook's own tabs, in its own order. One tab is not a choice, so a
          // single-sheet workbook gets its name instead.
          <div role="tablist" aria-label={S.sheets} className="flex gap-1 overflow-x-auto">
            {sheets.map((tab, i) => (
              <button
                key={tab.name}
                role="tab"
                aria-selected={i === active}
                onClick={() => setActive(i)}
                className={cn(
                  "shrink-0 rounded px-2 py-0.5 hover:bg-accent",
                  i === active && "bg-accent font-medium text-foreground",
                )}
              >
                {tab.name}
              </button>
            ))}
          </div>
        ) : (
          <span className="font-medium">{sheet.name}</span>
        )}
        <span className="shrink-0">
          {sheet.rows.length.toLocaleString()} {S.rows} · {sheet.widths.length} {S.columns}
          {sheet.truncated && ` · ${S.fileTooLarge}`}
        </span>
      </div>
    </div>
  );
}
