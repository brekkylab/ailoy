// A CSV reader for the file viewer.
//
// RFC 4180 is the spec being followed — quoted fields, `""` for a quote
// inside one, records that span lines — plus the two things real files do that the RFC
// does not mention: a delimiter that is not always a comma, and records that are
// ragged.
//
// Nothing here produces markup. The parser hands back strings and the viewer puts them
// in text nodes, so a cell whose text is `<script>` stays a cell whose text is
// `<script>`. There is no escaping step to get wrong because there is nothing to
// escape.

/// The delimiters worth sniffing for. A file that separates on something else is read
/// as one column per record, which is the honest answer rather than a wrong guess.
const DELIMITERS = [',', ';', '\t', '|'];

/// How much of the file the sniffer looks at. Enough records to tell a consistent
/// column count from a coincidence, cheap enough to parse twice.
const SNIFF_BYTES = 64 * 1024;
const SNIFF_ROWS = 40;

export interface Table {
  /// The delimiter the file was read with — sniffed unless one was given.
  delimiter: string;
  /// The first record. Treated as the header because that is the convention these
  /// files are written to; a file without one shows its first row of data up there.
  header: string[];
  /// Every record after the first, each padded to `columns`.
  rows: string[][];
  /// The width of the widest record. Ragged records are padded, never clipped: a
  /// trailing field that only one row has is still that row's data.
  columns: number;
}

/// The records of `src`, split on `delimiter`.
///
/// `limit` stops after that many records, for the sniffer. Blank lines are dropped —
/// in a one-column file an empty record and a blank line are the same bytes, and
/// skipping them is the reading that matches what the file means.
function split(src: string, delimiter: string, limit = Infinity): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  /// Whether a `"` here opens a quoted field rather than being a literal quote. Only
  /// true at the start of a field: `a"b` is three characters, not a broken quote.
  let start = true;
  let quoted = false;
  let i = 0;

  /// Ends the record, and reports whether `limit` has been reached.
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
      // Inside quotes every character is content, including the delimiter and the
      // newlines that make a record span lines. Only `"` ends it, and a doubled `"`
      // is one literal quote.
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

    // Text after a closing quote — `"a"b` — is appended rather than refused. The file
    // is malformed, but dropping the `b` would be a worse answer than keeping it.
    field += ch;
    start = false;
    i++;
  }

  // A file that does not end in a newline still ends a record, and so does one that
  // ends inside an unterminated quote.
  if (field !== '' || row.length) commit();
  return rows;
}

/// The delimiter `src` is most likely written with.
///
/// Each candidate is scored by how consistently it produces the same number of columns
/// across the sample, weighted by how many columns that is: a delimiter that splits
/// every record into 5 fields is a delimiter, and one that splits a few records
/// unevenly is a character that happens to appear in the text. Agreement is squared so
/// a clean 2-column read beats a ragged 3-column one.
function sniff(src: string): string {
  const sample = src.slice(0, SNIFF_BYTES);
  let best = DELIMITERS[0];
  let bestScore = 0;

  for (const delimiter of DELIMITERS) {
    const rows = split(sample, delimiter, SNIFF_ROWS);
    // The last record of a sliced sample may have been cut mid-line.
    const counts = (rows.length > 1 ? rows.slice(0, -1) : rows).map((row) => row.length);
    if (!counts.length) continue;

    const tally = new Map<number, number>();
    for (const n of counts) tally.set(n, (tally.get(n) ?? 0) + 1);

    let mode = 0;
    let hits = 0;
    for (const [n, count] of tally) {
      // Ties go to the wider read: 4 columns say more than 2.
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

/// `src` read as a table. `delimiter` overrides the sniffer — what a `.tsv` extension
/// already knows.
export function parseCsv(src: string, delimiter?: string): Table {
  // A BOM is a byte-order mark, not the first character of the first column name.
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

/// Whether a column holds numbers, and so should be read right-aligned.
///
/// Thousands separators and a leading currency symbol still count; a column of dates
/// or ids does not, which is why this is a per-column question and not a per-cell one.
/// Empty cells are ignored, and a column that is entirely empty is not numeric.
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
