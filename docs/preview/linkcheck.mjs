// Walks the built preview and fails on an internal link that does not resolve: a page that is
// not there, or a #fragment with no such id on its page. Taken from virtx's docs
// (docs/preview/linkcheck.mjs there), which run the same check.
//
//   node preview/linkcheck.mjs [DIST]   (default: dist)
//
// The preview serves the pages under /docs/, as the site does, so dist/ holds what the site has at
// /docs/. A link outside /docs/ is to one of the site's own pages, which the preview does not have:
// those in SITE_PAGES are let through, any other is broken.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const BASE = '/docs/';
const SITE_PAGES = [];
const dist = process.argv[2] ?? 'dist';

const walk = (dir) =>
  readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });

// file in dist → its ids and the href/src values on it
const pages = new Map();
for (const file of walk(dist).filter((f) => f.endsWith('.html'))) {
  const html = readFileSync(file, 'utf8');
  const ids = new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((m) => m[1]));
  const links = [...html.matchAll(/\s(?:href|src)="([^"]*)"/g)].map((m) => m[1].replaceAll('&amp;', '&'));
  pages.set(file, { ids, links });
}

// the file a path under /docs/ is served from, or undefined if there is none
function target(path) {
  const full = join(dist, path.slice(BASE.length));
  for (const f of path.endsWith('/') ? [join(full, 'index.html')] : [full, join(full, 'index.html')]) {
    try {
      if (statSync(f).isFile()) return f;
    } catch {}
  }
}

const broken = [];
let total = 0;
for (const [file, { links }] of pages) {
  for (const link of links) {
    if (!link.startsWith('/') || link.startsWith('//')) continue;
    total++;
    const [rawPath, frag] = link.split('#');
    const path = rawPath.replace(/\?.*$/, '');
    if (path && !path.startsWith(BASE)) {
      if (!SITE_PAGES.includes(path)) broken.push([file, link, 'outside /docs/']);
      continue;
    }
    const to = path ? target(path) : file;
    if (!to) broken.push([file, link, 'missing']);
    else if (frag && pages.has(to) && !pages.get(to).ids.has(frag)) broken.push([file, link, 'no such id']);
  }
}

console.log(`linkcheck: ${pages.size} pages, ${total} internal refs, ${broken.length} broken`);
for (const [file, link, why] of broken) console.log(`${relative(dist, file)}\t${link}\t${why}`);
process.exit(broken.length ? 1 : 0);
