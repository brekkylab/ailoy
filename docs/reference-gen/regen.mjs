// Regenerates the API reference from a release: npm run regen -- <tag>   (e.g. v0.3.2)
//
// Writes docs/content/reference/ (the shared Get started pages, python/ and nodejs/) and the reference groups at
// the end of docs/sidebar.json. Each language is read into one model (gen_py.py from the tag's
// stubs and the ailoy-py of that version, through uv; node_api.mjs from the @brekkylab/ailoy of
// that version), and render.mjs writes both from pages.mjs, so they share pages and layout. Rust is
// docs.rs, linked. Needs git, uv and the tag fetched.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PAGES, SECTIONS } from './pages.mjs';
import { EXT, example, render } from './render.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.resolve(HERE, '..');
const REFERENCE = path.join(DOCS, 'content', 'reference');
const SIDEBAR = path.join(DOCS, 'sidebar.json');
const SHARED = path.join(HERE, 'shared');
const EXAMPLES = path.join(HERE, 'examples');
const PY_TOOLS = ['griffe==2.4.1', 'black==26.10.1', 'mypy==1.18.2'];

const tag = process.argv[2];
if (!/^v\d+\.\d+\.\d+$/.test(tag ?? '')) fail('usage: npm run regen -- <tag>, such as v0.3.2');
const version = tag.slice(1);
if (!process.env.npm_execpath) fail('run it through npm: npm run regen -- ' + tag);

function fail(message) {
  console.error(`regen: ${message}`);
  process.exit(1);
}

function run(cmd, args, opts = {}) {
  return execFileSync(cmd, args, { stdio: ['ignore', 'pipe', 'inherit'], encoding: 'utf8', ...opts });
}

const npm = (args) => run(process.execPath, [process.env.npm_execpath, ...args], { cwd: HERE });

try {
  run('git', ['rev-parse', '--verify', '--quiet', `${tag}^{commit}`], { cwd: DOCS });
} catch {
  fail(`no tag ${tag} here: git fetch --tags`);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoy-reference-'));
const src = path.join(tmp, 'src');
run('git', ['worktree', 'add', '--detach', '--quiet', src, tag], { cwd: DOCS });

const unfilled = {};
const slugs = {};
try {
  // Node.js: the published package of this version, whose binding.d.ts napi writes at build
  npm(['ci', '--no-audit', '--no-fund']);
  npm(['install', '--no-save', '--no-audit', '--no-fund', `@brekkylab/ailoy@${version}`]);
  const nodeJson = path.join(tmp, 'nodejs.json');
  run(process.execPath, [path.join(HERE, 'node_api.mjs'), nodeJson], { cwd: HERE, stdio: 'inherit' });
  // every example, type-checked against the package it's shown for
  run(process.execPath, [path.join(HERE, 'node_modules', 'typescript', 'bin', 'tsc'), '-p', path.join(EXAMPLES, 'tsconfig.json')], { cwd: HERE, stdio: 'inherit' });

  // Python: the tag's stubs, with the docstrings of ailoy-py of this version
  const pyJson = path.join(tmp, 'python.json');
  const uvArgs = ['run', '--no-project', '--isolated', '--python', '3.12'];
  for (const p of [...PY_TOOLS, `ailoy-py==${version}`]) uvArgs.push('--with', p);
  run('uv', [...uvArgs, 'python', path.join(HERE, 'gen_py.py'), path.join(src, 'bindings', 'python', 'python'), pyJson, tag], { cwd: HERE });
  const pyExamples = fs.readdirSync(EXAMPLES).filter((f) => f.endsWith('.py')).map((f) => path.join(EXAMPLES, f));
  run('uv', [...uvArgs, 'mypy', '--strict', '--no-error-summary', ...pyExamples], { cwd: HERE, stdio: 'inherit' });

  for (const [lang, file] of [['python', pyJson], ['nodejs', nodeJson]]) {
    const api = JSON.parse(fs.readFileSync(file, 'utf8'));
    const out = render({ lang, api, version, tag, outDir: path.join(REFERENCE, lang) });
    slugs[lang] = out.slugs;
    unfilled[lang] = out.unfilled;
  }

  writeShared();
  writeSidebar(slugs);
} finally {
  run('git', ['worktree', 'remove', '--force', src], { cwd: DOCS });
  fs.rmSync(tmp, { recursive: true, force: true });
}

for (const [lang, keys] of Object.entries(unfilled)) {
  if (keys.length) console.warn(`regen: ${lang} has no description for ${keys.join(', ')}: add them to FILL in pages.mjs`);
}
console.log(`regen: wrote docs/content/reference/ and the reference groups in docs/sidebar.json from ${tag}.`);
console.log('regen: check it with npm run build in docs/, and look at the pages.');

// The pages shared by both languages (shared/*.mdx), with each `{/* example: name */}` as tabs
// of that example in every language it has.
function writeShared() {
  const tabs = (name) => {
    const items = [['python', 'Python', 'python'], ['nodejs', 'Node.js', 'javascript'], ['rust', 'Rust', 'rust']]
      .map(([lang, label, icon]) => [label, icon, EXT[lang], example(name, lang)])
      .filter(([, , , code]) => code)
      .map(([label, icon, ext, code]) => `<TabItem label="${label}" icon="seti:${icon}">\n\n\`\`\`${ext}\n${code}\n\`\`\`\n\n</TabItem>`);
    if (!items.length) fail(`shared pages: no example called ${name}`);
    return `<Tabs syncKey="lang">\n${items.join('\n')}\n</Tabs>`;
  };
  fs.rmSync(path.join(REFERENCE, 'index.md'), { force: true });
  for (const file of fs.readdirSync(SHARED)) {
    const text = fs.readFileSync(path.join(SHARED, file), 'utf8')
      .replace(/\{\/\* example: ([\w-]+) \*\/\}/g, (_, name) => tabs(name))
      .replaceAll('{{version}}', version);
    fs.writeFileSync(path.join(REFERENCE, file), text);
  }
}

// The reference groups go last in the sidebar, replacing the ones there: each language's SECTIONS,
// the same groups in the same order, its Get started opening with the shared pages. The
// docs' Sidebar.astro shows the API reference tab only the groups of the language being read, and
// the Docs tab none of them.
function writeSidebar(slugs) {
  const isReference = (gr) => gr.items.every((i) => i.slug?.startsWith('reference') || i.link?.startsWith('https://docs.rs/'));
  const groups = JSON.parse(fs.readFileSync(SIDEBAR, 'utf8')).filter((gr) => !isReference(gr));
  const titles = Object.fromEntries(PAGES.map((p) => [p.slug, p.title]));
  for (const lang of ['python', 'nodejs']) {
    for (const section of SECTIONS) {
      const items = section.pages
        .filter((s) => s === 'index' || slugs[lang].includes(s))
        .flatMap((s) =>
          s === 'index'
            ? [
                { slug: 'reference', label: 'Introduction' },
                { slug: 'reference/choosing-an-sdk', label: 'Choosing an SDK' },
                { slug: 'reference/choosing-a-model-provider', label: 'Choosing a model provider' },
                { slug: `reference/${lang}`, label: 'Installation' },
              ]
            : [{ slug: `reference/${lang}/${s}`, label: titles[s] }],
        );
      if (items.length) groups.push({ label: section.title, items });
    }
  }
  // the file's own layout: a group per block, a page per line
  const item = (o) => `{ ${Object.entries(o).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(', ')} }`;
  const group = (gr) =>
    `  {\n    "label": ${JSON.stringify(gr.label)},\n    "items": [\n${gr.items.map((i) => `      ${item(i)}`).join(',\n')}\n    ]\n  }`;
  fs.writeFileSync(SIDEBAR, `[\n${groups.map(group).join(',\n')}\n]\n`);
}
