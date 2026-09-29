// A CSV reader for the file viewer.
//
// Follows RFC 4180 (quoted fields, `""` for a quote, records spanning lines), plus two
// things real files do: non-comma delimiters and ragged records.
//
// Produces no markup: the viewer puts strings in text nodes, so a cell `<script>` stays
// text and there is nothing to escape.

/// Delimiters to sniff for. A file using anything else reads as one column per record
/// rather than as a wrong guess.
const DELIMITERS = [',', ';', '\t', '|'];

/// How much the sniffer samples: enough records to tell a consistent column count from a
/// coincidence, cheap enough to parse twice.
const SNIFF_BYTES = 64 * 1024;
const SNIFF_ROWS = 40;

export interface Table {
  /// Sniffed unless one was given.
  delimiter: string;
  /// The first record, by convention the header; a headerless file shows its first data row
  /// there.
  header: string[];
  /// Every record after the first, each padded to `columns`.
  rows: string[][];
  /// The widest record's width. Ragged records are padded, never clipped: a trailing field
  /// only one row has is still data.
  columns: number;
}

/// The records of `src`, split on `delimiter`.
///
/// `limit` caps the record count, for the sniffer. Blank lines are dropped: in a
/// one-column file an empty record and a blank line are the same bytes.
function split(src: string, delimiter: string, limit = Infinity): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  /// Whether a `"` here opens a quoted field. Only at the start of a field: `a"b` is three
  /// characters, not a broken quote.
  let start = true;
  let quoted = false;
  let i = 0;

  /// Ends the record; returns whether `limit` has been reached.
  const commit = (): boolean => {
    row.push(field);
    field = '';
    start = true;
    if (row.length > 1 || row[0] !== '') rows.push(row);
    row = [];
    return rows.length >= limit;
  };

  while (i < src.length) {
    const ch = src[i];

    if (quoted) {
      // Inside quotes everything is content, including delimiters and newlines; only `"` ends
      // it, and `""` is a literal quote.
      if (ch !== '"') {
        field += ch;
        i++;
      } else if (src[i + 1] === '"') {
        field += '"';
        i += 2;
      } else {
        quoted = false;
        i++;
      }
      continue;
    }

    if (start && ch === '"') {
      quoted = true;
      start = false;
      i++;
      continue;
    }

    if (ch === delimiter) {
      row.push(field);
      field = '';
      start = true;
      i++;
      continue;
    }

    if (ch === '\n' || ch === '\r') {
      const skip = ch === '\r' && src[i + 1] === '\n' ? 2 : 1;
      i += skip;
      if (commit()) return rows;
      continue;
    }

    // Text after a closing quote (`"a"b`) is kept rather than refused: malformed, but
    // dropping the `b` would be worse.
    field += ch;
    start = false;
    i++;
  }

  // End of input ends a record, even without a trailing newline or inside an unterminated
  // quote.
  if (field !== '' || row.length) commit();
  return rows;
}

/// The delimiter `src` is most likely written with.
///
/// Each candidate is scored by how consistently it yields the same column count across the
/// sample, weighted by that count; a character that splits a few records unevenly just
/// appears in the text. Agreement is squared so a clean 2-column read beats a ragged
/// 3-column one.
function sniff(src: string): string {
  const sample = src.slice(0, SNIFF_BYTES);
  let best = DELIMITERS[0];
  let bestScore = 0;

  for (const delimiter of DELIMITERS) {
    const rows = split(sample, delimiter, SNIFF_ROWS);
    // The last record of a sliced sample may be cut mid-line.
    const counts = (rows.length > 1 ? rows.slice(0, -1) : rows).map((row) => row.length);
    if (!counts.length) continue;

    const tally = new Map<number, number>();
    for (const n of counts) tally.set(n, (tally.get(n) ?? 0) + 1);

    let mode = 0;
    let hits = 0;
    for (const [n, count] of tally) {
      // Ties go to the wider read.
      if (count > hits || (count === hits && n > mode)) {
        mode = n;
        hits = count;
      }
    }
    if (mode < 2) continue;

    const agreement = hits / counts.length;
    const score = agreement * agreement * mode;
    if (score > bestScore) {
      bestScore = score;
      best = delimiter;
    }
  }

  return best;
}

/// `src` read as a table. `delimiter` overrides the sniffer, e.g. for `.tsv`.
export function parseCsv(src: string, delimiter?: string): Table {
  // Strip the BOM so it is not part of the first column name.
  const text = src.replace(/^﻿/, '');
  const sep = delimiter ?? sniff(text);
  const records = split(text, sep);
  const columns = records.reduce((wide, row) => Math.max(wide, row.length), 0);

  const pad = (row: string[]) =>
    row.length === columns ? row : row.concat(Array(columns - row.length).fill(''));

  return {
    delimiter: sep,
    header: pad(records[0] ?? []),
    rows: records.slice(1).map(pad),
    columns,
  };
}

/// Whether a column holds numbers, and so should be right-aligned.
///
/// Thousands separators and a leading currency symbol count; dates and ids do not, hence
/// per column rather than per cell. Empty cells are ignored; an all-empty column is not
/// numeric.
export function isNumericColumn(rows: string[][], index: number): boolean {
  let seen = 0;
  for (const row of rows) {
    const cell = (row[index] ?? '').trim();
    if (cell === '') continue;
    if (!/^[-+(]?[$€£¥₩]?\s?\d[\d,\s]*(?:\.\d+)?\)?%?$/.test(cell)) return false;
    seen++;
  }
  return seen > 0;
}
