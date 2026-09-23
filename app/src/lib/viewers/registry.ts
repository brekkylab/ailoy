// Which viewer opens which file.
//
// One table, keyed by extension, so adding a type is adding a component and a line
// here rather than a branch inside the Files view. A file whose extension is absent
// has no viewer, and the shell says so instead of guessing at the bytes.

import type { Component } from 'svelte';
import type { Entry } from './entry';
import type { Decoded } from '../encoding';
import CsvViewer from './CsvViewer.svelte';
import DocxViewer from './DocxViewer.svelte';
import HtmlViewer from './HtmlViewer.svelte';
import ImageViewer from './ImageViewer.svelte';
import MarkdownViewer from './MarkdownViewer.svelte';
import PdfViewer from './PdfViewer.svelte';
import XlsxViewer from './XlsxViewer.svelte';
import PlainTextViewer from './PlainTextViewer.svelte';
import { extOf } from '../files';

/// A viewer that is handed the file's text, already fetched and decoded.
export interface TextProps {
  entry: Entry;
  text: string;
  /// What a relative link inside the document resolves against, for the one viewer that
  /// has to answer it: the file's own folder in its tree (see `source.ts`).
  base: string;
  /// What those characters were decoded from. A viewer with room for it says so: the
  /// encoding of a file that did not announce one is something the app worked out, and
  /// a reader looking at Korean text should be able to see which way it was read.
  encoding: Decoded['encoding'];
}

/// A viewer that is handed the file's address and lets the browser do the fetching —
/// what a format wants when the renderer is the browser's own.
export interface UrlProps {
  entry: Entry;
  url: string;
}

/// A viewer that is handed the file's bytes, already fetched — what a format wants
/// when it is a container the app has to open itself rather than a stream of
/// characters or something the browser renders.
export interface BytesProps {
  entry: Entry;
  bytes: ArrayBuffer;
}

interface TextViewer {
  source: 'text';
  /// What the header calls the format.
  label: string;
  component: Component<TextProps>;
  /// Past this, the file is not opened inline. A viewer handed the text has to hold
  /// the whole of it in the DOM, and a browser asked to lay out a 40 MB document has
  /// stopped responding. `url` viewers stream and have no such limit.
  maxBytes: number;
}

interface UrlViewer {
  source: 'url';
  label: string;
  component: Component<UrlProps>;
}

interface BytesViewer {
  source: 'bytes';
  label: string;
  component: Component<BytesProps>;
  /// As with `text`: the whole file is held in memory and then turned into DOM, so
  /// there is a size past which it is not opened inline.
  maxBytes: number;
}

export type Viewer = TextViewer | UrlViewer | BytesViewer;

const MARKDOWN: TextViewer = {
  source: 'text',
  label: 'Markdown',
  component: MarkdownViewer,
  maxBytes: 2 * 1024 * 1024,
};

const TEXT: TextViewer = {
  source: 'text',
  label: 'Text',
  component: PlainTextViewer,
  // Between the prose viewers and the CSV one: a log is bigger than a document and
  // every line of it becomes a row, but the viewer pages through them, so what the cap
  // is really holding down is the string and the array it is split into.
  maxBytes: 2 * 1024 * 1024,
};

/// Same viewer, different name in the header. A `.log` is a text file that says what it
/// is, and the header saying `Log` is the one place that is worth repeating.
const LOG: TextViewer = { ...TEXT, label: 'Log' };

const HTML: TextViewer = {
  source: 'text',
  label: 'HTML',
  component: HtmlViewer,
  maxBytes: 4 * 1024 * 1024,
};

const CSV: TextViewer = {
  source: 'text',
  label: 'CSV',
  component: CsvViewer,
  // Lower than the prose viewers: a megabyte of CSV is tens of thousands of records,
  // and each one becomes a table row. The viewer pages through them, but the parse
  // itself still has to hold the whole sheet.
  maxBytes: 1024 * 1024,
};

/// Same viewer, different name in the header — and `CsvViewer` reads the extension
/// itself rather than sniffing a delimiter it already knows.
const TSV: TextViewer = { ...CSV, label: 'TSV' };

/// No `maxBytes`, like every `url` viewer: the bytes never pass through the app, and
/// a browser decoding a large photograph is a browser doing what it is built for.
const IMAGE: UrlViewer = {
  source: 'url',
  label: 'Image',
  component: ImageViewer,
};

/// The bytes and not the URL: pdf.js draws it (see `PdfViewer`), because the webview's
/// own PDF viewer shows nothing inside a frame here. Pages are drawn as they scroll into
/// view, so the cap is on the file held in memory, not on what gets laid out.
const PDF: BytesViewer = {
  source: 'bytes',
  label: 'PDF',
  component: PdfViewer,
  maxBytes: 100 * 1024 * 1024,
};

const DOCX: BytesViewer = {
  source: 'bytes',
  label: 'Word',
  component: DocxViewer,
  // A `.docx` is compressed, so the cap is on what arrives and not on what it becomes.
  // 25 MB of it is a document with a lot of photographs in it, which is the case worth
  // stopping: the pages it unpacks to are what the tab has to lay out.
  maxBytes: 25 * 1024 * 1024,
};

const XLSX: BytesViewer = {
  source: 'bytes',
  label: 'Spreadsheet',
  component: XlsxViewer,
  // Lower than the Word cap. A `.docx` that size is mostly pictures, which cost what
  // one image costs; an `.xlsx` that size is cells, and every one of them becomes a
  // table cell in the DOM.
  maxBytes: 12 * 1024 * 1024,
};

const BY_EXT: Record<string, Viewer | undefined> = {
  md: MARKDOWN,
  markdown: MARKDOWN,
  mdown: MARKDOWN,
  mkd: MARKDOWN,
  txt: TEXT,
  text: TEXT,
  log: LOG,
  csv: CSV,
  tsv: TSV,
  html: HTML,
  htm: HTML,
  png: IMAGE,
  jpg: IMAGE,
  jpeg: IMAGE,
  gif: IMAGE,
  webp: IMAGE,
  avif: IMAGE,
  bmp: IMAGE,
  ico: IMAGE,
  // Rendered in an <img>, which is the context that runs none of what makes the server
  // treat an SVG as active content.
  svg: IMAGE,
  pdf: PDF,
  docx: DOCX,
  docm: DOCX,
  xlsx: XLSX,
  xlsm: XLSX,
};

/// The viewer for `entry`, or null when nothing renders that type yet.
export function viewerFor(entry: Entry): Viewer | null {
  if (entry.type === 'folder') return null;
  return BY_EXT[extOf(entry.name)] ?? null;
}
