// A worksheet flattened into what the viewer draws.
//
// Converts a sheet (a sparse map of cells by row and column) into a table (a dense grid).
// Merges are the hard part: in the file a merge is a range with one cell holding the
// content; in a table it is a `colspan` on that cell and the absence of the cells it covers.
//
// Produces no markup: text comes back as strings for text nodes and styles as fixed fields
// (colour, weight, border), not CSS, so a workspace file reaches the DOM only as text.

import type { Cell as XCell, Row as XRow, Worksheet } from 'exceljs';

/// Past this the sheet is cut off and `truncated` says so. Forms run to tens of rows; a
/// hundred thousand is an export, and rendering it all hangs the tab.
const MAX_ROWS = 5000;
const MAX_COLUMNS = 128;

/// Excel measures column widths in characters of the default font and row heights in
/// points. Per Excel's file format notes, a character is 7 px plus 5 px padding, and a
/// point is 96/72 px.
const PX_PER_CHAR = 7;
const CELL_PADDING_PX = 5;
const PX_PER_POINT = 96 / 72;

/// Column width when the file sets none.
const DEFAULT_COLUMN_CHARS = 8.43;
/// Row height in points when neither the row nor the sheet sets one.
const DEFAULT_ROW_POINTS = 15;

export type Align = 'left' | 'center' | 'right';
export type VAlign = 'top' | 'middle' | 'bottom';

/// One side of a cell's border, reduced to the three things CSS needs.
export interface Border {
  width: number;
  style: 'solid' | 'dashed' | 'dotted' | 'double';
  color: string;
}

export interface Style {
  fill: string | null;
  color: string | null;
  bold: boolean;
  italic: boolean;
  underline: boolean;
  /// In pixels, or null to inherit the table's.
  size: number | null;
  family: string | null;
  align: Align | null;
  valign: VAlign | null;
  wrap: boolean;
  /// Absent sides are null, unlike a zero-width border: a neighbour's border still shows
  /// through.
  borders: { top: Border | null; right: Border | null; bottom: Border | null; left: Border | null };
}

export interface Cell {
  text: string;
  colSpan: number;
  rowSpan: number;
  style: Style;
}

export interface Row {
  /// In pixels. A minimum, not a fixed height, since a merged cell can make the row taller.
  height: number;
  cells: Cell[];
}

export interface Sheet {
  name: string;
  /// One per column, in pixels.
  widths: number[];
  rows: Row[];
  /// Set when the sheet was larger than the caps above.
  truncated: boolean;
}

// -- Colour ------------------------------------------------------
//
// A file colour is `AARRGGBB`, but the alpha byte is not alpha: Excel writes `00` on
// plainly opaque colours (the workspace's form fills are `00E8E8E8` and the like), so
// honouring it would make a grey header invisible. Only the last six digits are used.
//
// `theme` and `indexed` colours need the workbook theme to resolve; they are not resolved,
// and such cells render with no fill rather than a guessed colour.

interface ArgbColor {
  argb?: string;
  theme?: number;
  indexed?: number;
}

function cssColor(color: ArgbColor | undefined): string | null {
  const argb = color?.argb;
  if (typeof argb !== 'string' || !/^[0-9a-f]{8}$/i.test(argb)) return null;
  return `#${argb.slice(2)}`;
}

// -- Borders -----------------------------------------------------

/// Excel names a border's weight and pattern together; CSS wants them apart.
const BORDER_STYLES: Record<string, { width: number; style: Border['style'] }> = {
  hair: { width: 1, style: 'solid' },
  thin: { width: 1, style: 'solid' },
  medium: { width: 2, style: 'solid' },
  thick: { width: 3, style: 'solid' },
  double: { width: 3, style: 'double' },
  dotted: { width: 1, style: 'dotted' },
  dashed: { width: 1, style: 'dashed' },
  dashDot: { width: 1, style: 'dashed' },
  dashDotDot: { width: 1, style: 'dashed' },
  mediumDashed: { width: 2, style: 'dashed' },
  mediumDashDot: { width: 2, style: 'dashed' },
  mediumDashDotDot: { width: 2, style: 'dashed' },
  slantDashDot: { width: 1, style: 'dashed' },
};

/// Fallback border colour: a border the file asked for is drawn even without a colour,
/// or the grid would be lost.
const DEFAULT_BORDER_COLOR = '#9a9a9a';

function borderOf(side: { style?: string; color?: ArgbColor } | undefined): Border | null {
  if (!side?.style) return null;
  const shape = BORDER_STYLES[side.style] ?? { width: 1, style: 'solid' as const };
  return { ...shape, color: cssColor(side.color) ?? DEFAULT_BORDER_COLOR };
}

// -- Values ------------------------------------------------------

/// A number format, reduced to what it does to a number. Only the first section is read;
/// the `;` sections (negative, zero, text) are ignored, which a preview can afford.
interface NumberFormat {
  decimals: number;
  grouped: boolean;
  percent: boolean;
}

function parseNumberFormat(fmt: string): NumberFormat {
  const positive = fmt.split(';')[0];
  const decimals = /\.(0+)/.exec(positive);
  return {
    decimals: decimals ? decimals[1].length : 0,
    grouped: positive.includes(','),
    percent: positive.includes('%'),
  };
}

/// Whether a format is a date. `m` means minutes or months depending on context, so the
/// test uses only unambiguous parts.
function isDateFormat(fmt: string): boolean {
  return /y{2,}|d{1,2}|h{1,2}|s{1,2}/i.test(fmt.replace(/\[[^\]]*\]/g, '').replace(/"[^"]*"/g, ''));
}

function formatNumber(value: number, fmt: string | undefined): string {
  if (!fmt || fmt === 'General') {
    // A spreadsheet's display of an unformatted number, without exponent notation for large
    // ones.
    return String(value);
  }
  const { decimals, grouped, percent } = parseNumberFormat(fmt);
  const n = percent ? value * 100 : value;
  const text = n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: grouped,
  });
  return percent ? `${text}%` : text;
}

function pad(n: number, width = 2): string {
  return String(n).padStart(width, '0');
}

/// Date tokens, substituted longest first so `yyyy` is not read as two `yy`. Separators
/// and literal text are left as is.
function formatDate(value: Date, fmt: string | undefined): string {
  if (!fmt) return value.toISOString().slice(0, 10);
  return fmt
    .replace(/\[[^\]]*\]/g, '')
    .replace(/yyyy/gi, String(value.getFullYear()))
    .replace(/yy/gi, pad(value.getFullYear() % 100))
    .replace(/mmmm/g, value.toLocaleString('en-US', { month: 'long' }))
    .replace(/mmm/g, value.toLocaleString('en-US', { month: 'short' }))
    .replace(/dddd/gi, value.toLocaleString('en-US', { weekday: 'long' }))
    .replace(/ddd/gi, value.toLocaleString('en-US', { weekday: 'short' }))
    .replace(/dd/gi, pad(value.getDate()))
    .replace(/d/gi, String(value.getDate()))
    .replace(/hh/g, pad(value.getHours()))
    .replace(/h/g, String(value.getHours()))
    .replace(/ss/g, pad(value.getSeconds()))
    .replace(/mm/g, pad(value.getMonth() + 1))
    .replace(/m/g, String(value.getMonth() + 1))
    .replace(/"/g, '')
    .trim();
}

/// Any cell value as display text.
///
/// Formulas show their cached result, never evaluated. A blank result (as in an empty
/// form) stays blank.
function textOf(value: unknown, numFmt: string | undefined): string {
  if (value == null) return '';
  if (value instanceof Date) return formatDate(value, numFmt);
  if (typeof value === 'number') {
    return numFmt && isDateFormat(numFmt)
      ? formatNumber(value, undefined)
      : formatNumber(value, numFmt);
  }
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'TRUE' : 'FALSE';
  if (typeof value !== 'object') return String(value);

  const record = value as Record<string, unknown>;
  if ('error' in record) return String(record.error);
  if ('formula' in record || 'sharedFormula' in record) return textOf(record.result, numFmt);
  if ('richText' in record) {
    const runs = record.richText as { text?: string }[] | undefined;
    return (runs ?? []).map((run) => run.text ?? '').join('');
  }
  // A hyperlink cell. The address is not rendered as a link: a preview click must not
  // navigate out of the app.
  if ('text' in record) return String(record.text);
  if ('hyperlink' in record) return String(record.hyperlink);
  return '';
}

// -- The grid ----------------------------------------------------

const ALIGNMENTS: Record<string, Align> = {
  left: 'left',
  center: 'center',
  centerContinuous: 'center',
  right: 'right',
  justify: 'left',
  fill: 'left',
  distributed: 'center',
};

const VERTICAL: Record<string, VAlign> = {
  top: 'top',
  middle: 'middle',
  bottom: 'bottom',
  distributed: 'middle',
  justify: 'middle',
};

function styleOf(cell: XCell, numeric: boolean): Style {
  const font = cell.font ?? {};
  const alignment = cell.alignment ?? {};
  const border = cell.border ?? {};
  // `fgColor`, not `bgColor`: in a solid pattern fill the foreground is the colour block.
  const fill = cell.fill?.type === 'pattern' && cell.fill.pattern === 'solid'
    ? cssColor(cell.fill.fgColor)
    : null;

  return {
    fill,
    color: cssColor(font.color),
    bold: font.bold === true,
    italic: font.italic === true,
    underline: font.underline !== undefined && font.underline !== false,
    size: typeof font.size === 'number' ? font.size * PX_PER_POINT : null,
    family: typeof font.name === 'string' ? font.name : null,
    // Unformatted numbers align right, as in a spreadsheet, so figure columns read cleanly.
    align: ALIGNMENTS[alignment.horizontal ?? ''] ?? (numeric ? 'right' : null),
    valign: VERTICAL[alignment.vertical ?? ''] ?? null,
    wrap: alignment.wrapText === true,
    borders: {
      top: borderOf(border.top),
      right: borderOf(border.right),
      bottom: borderOf(border.bottom),
      left: borderOf(border.left),
    },
  };
}

/// How far a merge starting at (row, column) reaches.
///
/// Found by walking out from the master, since the worksheet's merge list is not public:
/// every covered cell reports the master as its own, so the edge is the first that does not.
function spanOf(sheet: Worksheet, master: XCell, row: number, column: number, rows: number, columns: number) {
  let colSpan = 1;
  while (column + colSpan <= columns && sheet.getCell(row, column + colSpan).master === master) {
    colSpan++;
  }
  let rowSpan = 1;
  while (row + rowSpan <= rows && sheet.getCell(row + rowSpan, column).master === master) {
    rowSpan++;
  }
  return { colSpan, rowSpan };
}

function heightOf(row: XRow, fallback: number): number {
  const points = typeof row.height === 'number' ? row.height : fallback;
  return Math.round(points * PX_PER_POINT);
}

/// One worksheet as a grid of rows.
export function readSheet(sheet: Worksheet): Sheet {
  const rowCount = Math.min(sheet.rowCount, MAX_ROWS);
  const columnCount = Math.min(sheet.columnCount, MAX_COLUMNS);
  const truncated = sheet.rowCount > rowCount || sheet.columnCount > columnCount;

  const widths: number[] = [];
  for (let c = 1; c <= columnCount; c++) {
    const chars = sheet.getColumn(c).width ?? DEFAULT_COLUMN_CHARS;
    widths.push(Math.round(chars * PX_PER_CHAR + CELL_PADDING_PX));
  }

  const defaultPoints = sheet.properties?.defaultRowHeight ?? DEFAULT_ROW_POINTS;
  const rows: Row[] = [];

  for (let r = 1; r <= rowCount; r++) {
    const source = sheet.getRow(r);
    const cells: Cell[] = [];

    for (let c = 1; c <= columnCount; c++) {
      const cell = sheet.getCell(r, c);
      // A covered cell is not a table cell: the master spans it, and emitting it would push
      // the rest of the row over a column.
      if (cell.master !== cell) continue;

      const { colSpan, rowSpan } = spanOf(sheet, cell, r, c, rowCount, columnCount);
      const text = textOf(cell.value, cell.numFmt);
      // Checked on the value, not the text, so a figure formatted with a currency symbol still
      // counts.
      const numeric = typeof cell.value === 'number' || cell.value instanceof Date;

      cells.push({ text, colSpan, rowSpan, style: styleOf(cell, numeric) });
    }

    rows.push({ height: heightOf(source, defaultPoints), cells });
  }

  return { name: sheet.name, widths, rows, truncated };
}
