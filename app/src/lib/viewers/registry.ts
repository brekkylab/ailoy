// Which viewer opens which file.
//
// One table keyed by extension, so a new type is a component plus a line here, not a
// branch in the Files view. An unlisted extension has no viewer, and the shell says so
// instead of guessing at the bytes.

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
  /// What relative links in the document resolve against: the file's own folder in its tree
  /// (see `source.ts`).
  base: string;
  /// The encoding the text was decoded from. Viewers with room show it, since for a file
  /// that did not announce one it was inferred, and a reader of Korean text should see how.
  encoding: Decoded['encoding'];
}

/// A viewer handed the file's URL, for formats the browser renders itself.
export interface UrlProps {
  entry: Entry;
  url: string;
}

/// A viewer handed the file's bytes, for container formats the app opens itself.
export interface BytesProps {
  entry: Entry;
  bytes: ArrayBuffer;
}

interface TextViewer {
  source: 'text';
  /// What the header calls the format.
  label: string;
  component: Component<TextProps>;
  /// Past this, the file is not opened inline.
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
  /// Past this, the file is not opened inline.
  maxBytes: number;
}

/// `text` and `bytes` viewers hold the whole file in memory and turn it into DOM (a large
/// document hangs the browser), so each caps inline size with `maxBytes`. `url` viewers
/// let the browser fetch and stream, and have no cap.
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
  // Each line becomes a row; the viewer pages through them, so the cap bounds the string
  // and the array it is split into.
  maxBytes: 2 * 1024 * 1024,
};

/// Plain text labeled `Log` in the header.
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
  // A megabyte of CSV is tens of thousands of table rows; the viewer pages them, but the
  // parse holds the whole sheet.
  maxBytes: 1024 * 1024,
};

/// `CsvViewer` reads the extension itself rather than sniffing a delimiter it knows.
const TSV: TextViewer = { ...CSV, label: 'TSV' };

/// No `maxBytes`: the bytes never pass through the app.
const IMAGE: UrlViewer = {
  source: 'url',
  label: 'Image',
  component: ImageViewer,
};

/// Bytes, not URL: pdf.js draws it (see `PdfViewer`), because the webview's own PDF viewer
/// shows nothing inside a frame here. Pages render as they scroll into view, so the cap is
/// on the file held in memory, not on layout.
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
  // Compressed, so the cap is on the download, not the unpacked size. A large one is mostly
  // photographs, and the unpacked pages are what the tab must lay out.
  maxBytes: 25 * 1024 * 1024,
};

const XLSX: BytesViewer = {
  source: 'bytes',
  label: 'Spreadsheet',
  component: XlsxViewer,
  // An `.xlsx` this size is mostly cells, each of which becomes a DOM table cell.
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
  // Rendered in an <img>, which runs none of an SVG's active content.
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
