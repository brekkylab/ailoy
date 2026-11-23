// Regenerates the API reference from a release: npm run regen -- <tag>   (e.g. v0.3.2)
//
// Writes docs/content/reference/ (the overview, python/ and nodejs/) and the reference groups at
// the end of docs/sidebar.json. Python comes from the tag's stubs and the ailoy-py of that version
// (gen_py.py, through uv); Node.js from the @brekkylab/ailoy of that version (typedoc). Rust is
// docs.rs, linked. Needs git, uv and the tag fetched.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.resolve(HERE, '..');
const REFERENCE = path.join(DOCS, 'content', 'reference');
const SIDEBAR = path.join(DOCS, 'sidebar.json');
const PY_TOOLS = ['griffe==2.4.1', 'griffe2md==1.5.1', 'black==26.10.1'];

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

try {
  // Node.js: the published package of this version, whose binding.d.ts napi writes at build
  npm(['ci', '--no-audit', '--no-fund']);
  npm(['install', '--no-save', '--no-audit', '--no-fund', `@brekkylab/ailoy@${version}`]);
  const nodeSidebar = path.join(tmp, 'nodejs-sidebar.json');
  run(process.execPath, [path.join(HERE, 'node_modules', 'typedoc', 'bin', 'typedoc'), '--logLevel', 'Warn'], {
    cwd: HERE,
    env: { ...process.env, AILOY_REF_TAG: tag, AILOY_REF_SIDEBAR_OUT: nodeSidebar },
    stdio: 'inherit',
  });
  const node = JSON.parse(fs.readFileSync(nodeSidebar, 'utf8'));

  // Python: the tag's stubs, with the docstrings of ailoy-py of this version
  const pyOut = path.join(REFERENCE, 'python');
  fs.rmSync(pyOut, { recursive: true, force: true });
  const uvArgs = ['run', '--no-project', '--isolated', '--python', '3.12'];
  for (const p of [...PY_TOOLS, `ailoy-py==${version}`]) uvArgs.push('--with', p);
  const pySlugs = run('uv', [...uvArgs, 'python', path.join(HERE, 'gen_py.py'),
    path.join(src, 'bindings', 'python', 'python'), pyOut, tag], { cwd: HERE }).trim().split(/\s+/);

  fs.writeFileSync(path.join(REFERENCE, 'index.md'), overview());
  writeSidebar(pySlugs.map((s) => `reference/python/${s}`), [...node.classes, ...node.functions]);
} finally {
  run('git', ['worktree', 'remove', '--force', src], { cwd: DOCS });
  fs.rmSync(tmp, { recursive: true, force: true });
}

console.log(`regen: wrote docs/content/reference/ and the reference groups in docs/sidebar.json from ${tag}.`);
console.log('regen: check it with npm run build in docs/, and look at the pages.');

function overview() {
  const rs = `https://docs.rs/ailoy/${version}/ailoy`;
  return `---
title: API reference
description: "The Ailoy SDK reference for Python, Node.js and Rust, generated from the ${tag} release."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

Every class, function and type the Ailoy SDK exports, per language. It's generated from the [${tag}](https://github.com/brekkylab/ailoy/tree/${tag}) release, so it describes the published packages. The guides follow \`main\`, which can be ahead of the release.

| Language | Package | Reference |
| --- | --- | --- |
| Python | \`ailoy-py\` ${version} | [Python API reference](/docs/reference/python/) |
| Node.js | \`@brekkylab/ailoy\` ${version} | [Node.js API reference](/docs/reference/nodejs/) |
| Rust | \`ailoy\` ${version} | [docs.rs/ailoy/${version}](${rs}/) |

Where to start, in any language:

- \`AgentBuilder\`: sets up an agent and builds it. [Python](/docs/reference/python/agent-builder/), [Node.js](/docs/reference/nodejs/agent-builder/), [Rust](${rs}/agent/struct.AgentBuilder.html)
- \`Agent\`: runs turns and holds the history. [Python](/docs/reference/python/agent/), [Node.js](/docs/reference/nodejs/agent/), [Rust](${rs}/agent/struct.Agent.html)
- \`LangModel\`: a model run without an agent. [Python](/docs/reference/python/lang-model/), [Node.js](/docs/reference/nodejs/lang-model/)
`;
}

// The reference groups go last in the sidebar, replacing the ones there. The docs' Sidebar.astro
// shows two levels, a group and its pages, so each language is a group of its own.
function writeSidebar(python, nodejs) {
  const groups = JSON.parse(fs.readFileSync(SIDEBAR, 'utf8'))
    .filter((g) => !['Reference', 'Python reference', 'Node.js reference'].includes(g.label));
  groups.push(
    { label: 'Reference', items: [{ slug: 'reference' }, { label: 'Rust (docs.rs)', link: `https://docs.rs/ailoy/${version}/ailoy/` }] },
    { label: 'Python reference', items: [{ slug: 'reference/python', label: 'Overview' }, ...python.map((slug) => ({ slug }))] },
    { label: 'Node.js reference', items: [{ slug: 'reference/nodejs', label: 'Overview' }, ...nodejs.map((slug) => ({ slug }))] },
  );
  // the file's own layout: a group per block, a page per line
  const item = (o) => `{ ${Object.entries(o).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(', ')} }`;
  const group = (g) =>
    `  {\n    "label": ${JSON.stringify(g.label)},\n    "items": [\n${g.items.map((i) => `      ${item(i)}`).join(',\n')}\n    ]\n  }`;
  fs.writeFileSync(SIDEBAR, `[\n${groups.map(group).join(',\n')}\n]\n`);
}
