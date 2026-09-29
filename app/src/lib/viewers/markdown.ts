// A Markdown renderer for the file viewer.
//
// The CommonMark core plus GFM tables, which is what a preview needs — not the whole
// spec. What is left out on purpose: setext headings (`===` under a line, which `---` makes ambiguous
// against a rule), reference links, and raw HTML.
//
// Raw HTML is not "unsupported" so much as refused. The source is a file from the
// workspace, which is a file someone else may have written, so every character is
// escaped *before* it is parsed. Nothing in the input can reach the DOM as markup;
// only the tags this module writes itself can.

/// The characters that change meaning in HTML text and in an attribute value.
const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

/// A URL that is safe to put in `href`/`src`, or null when it is not one.
///
/// Relative and fragment links pass, and so do the three schemes a document has a
/// reason to use. Anything else carrying a scheme — `javascript:` being the one that
/// matters — is dropped, and the link renders as the literal text it was written as.
function safeUrl(raw: string): string | null {
  const url = raw.trim();
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) && !/^(?:https?|mailto|tel):/i.test(url)) return null;
  return url;
}

// -- Inline ------------------------------------------------------
//
// Inline rendering runs on text that is already escaped, so the patterns below only
// ever see `&lt;` where the source had `<`. Finished fragments -- a code span, an <a>
// tag -- are parked in `slots` behind a sentinel so later passes cannot reach inside
// them: emphasis must not fire on a `*` that lives in a code span or a URL.

/// NUL, which decoded UTF-8 text from the workspace will not contain.
const SENTINEL = String.fromCharCode(0);
const PARKED = new RegExp(`${SENTINEL}(\\d+)${SENTINEL}`, 'g');

function inline(src: string): string {
  const slots: string[] = [];
  const park = (html: string) => `${SENTINEL}${slots.push(html) - 1}${SENTINEL}`;

  let out = escapeHtml(src);

  // Two or more trailing spaces are a hard break. Handled first, because the paragraph
  // that owns these lines has already joined them with newlines.
  out = out.replace(/ {2,}\n/g, () => park('<br />'));

  // Code spans before everything else, backslash escapes included: inside a code span
  // a backslash is a backslash.
  out = out.replace(/(`+)([^`]|[^`][\s\S]*?)\1(?!`)/g, (_m, _ticks: string, code: string) =>
    park(`<code>${code.replace(/^ (.*) $/, '$1')}</code>`),
  );

  out = out.replace(/\\([\\`*_{}[\]()#+\-.!>~|])/g, (_m, ch: string) => park(escapeHtml(ch)));

  // Images before links: the syntaxes differ only by the leading `!`.
  out = out.replace(
    /!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"([^"]*)")?\s*\)/g,
    (whole: string, alt: string, href: string, title?: string) => {
      const src = safeUrl(href);
      if (!src) return whole;
      return park(`<img src="${src}" alt="${alt}"${title ? ` title="${title}"` : ''} />`);
    },
  );

  // Only the tags are parked. The link text stays in the stream, so emphasis and code
  // inside it still render.
  out = out.replace(
    /\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"([^"]*)")?\s*\)/g,
    (whole: string, text: string, href: string, title?: string) => {
      const url = safeUrl(href);
      if (!url) return whole;
      const open =
        `<a href="${url}"${title ? ` title="${title}"` : ''}` +
        ' target="_blank" rel="noreferrer noopener">';
      return park(open) + text + park('</a>');
    },
  );

  out = out.replace(/&lt;((?:https?:\/\/|mailto:)[^\s]*?)&gt;/g, (_m, url: string) =>
    park(`<a href="${url}" target="_blank" rel="noreferrer noopener">${url}</a>`),
  );

  out = out.replace(/~~(\S|\S[\s\S]*?\S)~~/g, '<del>$1</del>');
  out = out.replace(/\*\*(\S|\S[\s\S]*?\S)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[\s(])__(\S|\S[\s\S]*?\S)__(?=$|[\s).,;:!?])/g, '$1<strong>$2</strong>');
  out = out.replace(/\*(\S|\S[\s\S]*?\S)\*/g, '<em>$1</em>');
  // `_` only at a word boundary: snake_case names are common in these files, and
  // `lead_time_days` is not emphasis.
  out = out.replace(/(^|[\s(])_(\S|\S[\s\S]*?\S)_(?=$|[\s).,;:!?])/g, '$1<em>$2</em>');

  return out.replace(PARKED, (_m, i: string) => slots[Number(i)]);
}

// -- Blocks ------------------------------------------------------

const FENCE = /^ {0,3}(```+|~~~+)\s*([^`\s]*)/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^ {0,3}([-*_])\s*(?:\1\s*){2,}$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
/// The `|---|:--:|` line that turns the row above it into a table header.
const TABLE_RULE = /^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$/;

/// A cell boundary is an unescaped pipe; `\|` is a pipe inside a cell.
const CELL_SPLIT = /(?<!\\)\|/;

/// True when the line starts a block that a paragraph must not swallow.
function startsBlock(line: string): boolean {
  return (
    line.trim() === '' ||
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    ITEM.test(line)
  );
}

function blocks(lines: string[]): string {
  const out: string[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === '') {
      i++;
      continue;
    }

    const fence = line.match(FENCE);
    if (fence) {
      const close = new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
      const body: string[] = [];
      i++;
      while (i < lines.length && !close.test(lines[i])) body.push(lines[i++]);
      i++; // the closing fence, or the end of the file when it is missing
      const lang = fence[2] ? ` class="lang-${escapeHtml(fence[2])}"` : '';
      out.push(`<pre><code${lang}>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      i++;
      continue;
    }

    if (RULE.test(line)) {
      out.push('<hr />');
      i++;
      continue;
    }

    if (QUOTE.test(line)) {
      const body: string[] = [];
      // A quote runs until a blank line. Lines without their own `>` belong to it too,
      // which is how a wrapped quoted paragraph is written.
      while (i < lines.length && lines[i].trim() !== '') {
        const quoted = lines[i].match(QUOTE);
        if (!quoted && startsBlock(lines[i])) break;
        body.push(quoted ? quoted[1] : lines[i]);
        i++;
      }
      out.push(`<blockquote>${blocks(body)}</blockquote>`);
      continue;
    }

    if (ITEM.test(line)) {
      const [html, next] = takeList(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
      const [html, next] = takeTable(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    const para: string[] = [];
    while (i < lines.length && !startsBlock(lines[i])) {
      if (lines[i].includes('|') && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) break;
      para.push(lines[i++]);
    }
    out.push(`<p>${inline(para.join('\n'))}</p>`);
  }

  return out.join('\n');
}

/// One list, and the line after it.
///
/// Nesting is not handled here: an indented line is content of the item above it, so a
/// nested list is found by `blocks` when the item's own lines are parsed.
function takeList(lines: string[], start: number): [string, number] {
  const first = lines[start].match(ITEM)!;
  const base = first[1].length;
  const ordered = /\d/.test(first[2]);

  const items: string[][] = [];
  let loose = false;
  let blanks = 0;
  let i = start;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === '') {
      blanks++;
      i++;
      continue;
    }

    const item = line.match(ITEM);
    const indent = line.length - line.trimStart().length;

    if (item && indent <= base + 1 && /\d/.test(item[2]) === ordered) {
      // A blank line before a sibling makes the list loose, so its items get <p>.
      if (blanks && items.length) loose = true;
      blanks = 0;
      items.push([item[3]]);
      i++;
      continue;
    }

    if (!items.length) break;

    if (indent > base + 1) {
      // Continuation: dedented by the marker's width so a nested list reads as a list
      // at column zero when the item's lines are parsed on their own.
      if (blanks) loose = true;
      for (; blanks > 0; blanks--) items[items.length - 1].push('');
      items[items.length - 1].push(line.slice(Math.min(indent, base + 2)));
      i++;
      continue;
    }

    // Unindented and not an item: part of the item's paragraph when it follows it
    // directly, and the end of the list when a blank line came between.
    if (blanks || startsBlock(line)) break;
    items[items.length - 1].push(line);
    i++;
  }

  const rendered = items
    .map((item) => {
      const html = blocks(item);
      // A tight item wears no <p>, including when a nested list follows its first
      // paragraph — only that leading paragraph is unwrapped.
      const tight = loose ? html : html.replace(/^<p>((?:(?!<\/p>)[\s\S])*)<\/p>/, '$1');
      return `<li>${tight}</li>`;
    })
    .join('\n');

  if (!ordered) return [`<ul>\n${rendered}\n</ul>`, i];
  const from = Number(first[2].slice(0, -1));
  return [`<ol${from === 1 ? '' : ` start="${from}"`}>\n${rendered}\n</ol>`, i];
}

/// One GFM pipe table, and the line after it.
function takeTable(lines: string[], start: number): [string, number] {
  const cells = (row: string) =>
    row
      .trim()
      .replace(/^\||\|$/g, '')
      .split(CELL_SPLIT)
      .map((cell) => cell.trim());

  const head = cells(lines[start]);
  const align = cells(lines[start + 1]).map((spec) => {
    const left = spec.startsWith(':');
    const right = spec.endsWith(':');
    if (left && right) return ' style="text-align:center"';
    if (right) return ' style="text-align:right"';
    if (left) return ' style="text-align:left"';
    return '';
  });

  let i = start + 2;
  const body: string[][] = [];
  while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
    body.push(cells(lines[i]));
    i++;
  }

  const th = head.map((cell, n) => `<th${align[n] ?? ''}>${inline(cell)}</th>`).join('');
  const rows = body
    .map((row) => {
      // Rows are padded or clipped to the header: a ragged row should not shift the
      // columns under it.
      const tds = head
        .map((_, n) => `<td${align[n] ?? ''}>${inline(row[n] ?? '')}</td>`)
        .join('');
      return `<tr>${tds}</tr>`;
    })
    .join('\n');

  return [`<table><thead><tr>${th}</tr></thead><tbody>\n${rows}\n</tbody></table>`, i];
}

/// `src` as HTML, safe to insert as markup.
export function renderMarkdown(src: string): string {
  const lines = src
    .replace(/^﻿/, '')
    .replace(/\r\n?/g, '\n')
    // Tabs are indentation here, and the list parser counts columns.
    .replace(/\t/g, '    ')
    .split('\n');
  return blocks(lines);
}
