// Writes a language's reference pages from its API model (gen_py.py, node_api.mjs) and pages.mjs.
//
// Every page has the same anatomy in both languages: the summary and what the page covers, the
// example (examples/<slug>.py or .js, type-checked by regen) and the guides, a table of what is on
// it, then each name: what it does,
// its signature, its parameters, what it returns and what it raises.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FILL, PAGES, RAISES, SECTIONS } from './pages.mjs';

const EXAMPLES = fileURLToPath(new URL('examples', import.meta.url));
export const EXT = { python: 'py', nodejs: 'js', rust: 'rs' };
/** The example `name` in `lang`, or '' when it has none. */
export function example(name, lang) {
  const file = path.join(EXAMPLES, `${name}.${EXT[lang]}`);
  return fs.existsSync(file) ? fs.readFileSync(file, 'utf8').trimEnd() : '';
}

export const LANGS = {
  python: { label: 'Python', pkg: 'ailoy-py', fence: 'python', null: 'None', true: 'True', false: 'False' },
  nodejs: { label: 'Node.js', pkg: '@brekkylab/ailoy', fence: 'ts', null: 'null', true: 'true', false: 'false' },
};

const snake = (s) => s.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
const isTypeName = (s) => /^[A-Z]/.test(s);

/** Text for `lang`: a string or { python, nodejs }, with {{name}} as the language spells it. */
function say(value, lang) {
  if (value == null) return '';
  const text = typeof value === 'string' ? value : value[lang] ?? '';
  return text.replace(/\{\{(\w+)(\(\))?\}\}/g, (_, n, call = '') => {
    if (n === 'None') return `\`${LANGS[lang].null}\``;
    return `\`${lang === 'nodejs' && !isTypeName(n) ? camel(n) : n}${call}\``;
  });
}

// the doc comments come from Rust and Python docstrings: rustdoc links, RST literals, em dashes
function clean(text, version) {
  return (text ?? '')
    .replace(/``([^`]+)``/g, '`$1`')
    .replace(/\[`crate::([\w:]+)`\](?!\()/g, (_, p) => `[\`crate::${p}\`](https://docs.rs/ailoy/${version}/ailoy/${p.replace(/::/g, '/')}/index.html)`)
    .replace(/\[(`[^`\]]+`)\](?!\()/g, '$1')
    .replace(/\s*—\s*/g, ', ')
    .trim();
}

const firstSentence = (text) => {
  const flat = text.replace(/\s+/g, ' ').trim();
  const m = flat.match(/^.*?[.!?](?=\s|$)/);
  return m ? m[0] : flat;
};

// the heading's id as Starlight makes it (github-slugger)
const slug = (s) => s.toLowerCase().replace(/[^\w\s-]/g, '').trim().replace(/\s/g, '-');
const cell = (s) => s.replace(/\|/g, '\\|').replace(/\n+/g, ' ');

export function render({ lang, api, version, tag, outDir }) {
  const L = LANGS[lang];
  const canon = (name) => (lang === 'nodejs' && !isTypeName(name) ? snake(name) : name);
  const own = (canonical) => (lang === 'nodejs' && !isTypeName(canonical) ? camel(canonical) : canonical);

  // where each name is documented, for links from types
  const where = new Map();
  const placed = new Set();
  for (const page of PAGES) {
    for (const item of page.items) {
      const name = own(item);
      if (!api[name]) continue;
      if (placed.has(name)) throw new Error(`render: ${name} is on two pages`);
      placed.add(name);
      // a class that is its page's whole subject has no heading of its own: link the page
      const whole = page.items.length === 1 && api[name].kind === 'class';
      where.set(name, `/docs/reference/${lang}/${page.slug}/${whole ? '' : `#${name.toLowerCase()}`}`);
    }
  }
  const missing = Object.keys(api).filter((n) => !placed.has(n));
  if (missing.length) throw new Error(`render: ${L.label} exports ${missing.join(', ')}, on no page in pages.mjs`);

  const unfilled = [];
  const fill = (key) => FILL[key] ?? {};
  const docFor = (key, doc) => {
    const f = typeof FILL[key] === 'string' ? { doc: FILL[key] } : fill(key);
    // `override` replaces a doc comment that reads as the binding's internals
    const own = f.override ? say(f.doc, lang) : clean(doc, version) || say(f.doc, lang);
    const parts = [own, say(f.more, lang)].filter(Boolean);
    if (!parts.length) unfilled.push(key);
    return parts.join('\n\n');
  };

  // a type as code, with each name the reference documents linked
  const typeMd = (t) => {
    if (!t) return '';
    const out = [];
    let code = '';
    for (const tok of t.replace(/\s+/g, ' ').split(/(\b[A-Z]\w*\b)/)) {
      if (where.has(tok)) {
        if (code) out.push(`\`${code}\``);
        code = '';
        out.push(`[\`${tok}\`](${where.get(tok)})`);
      } else code += tok;
    }
    if (code.trim()) out.push(`\`${code}\``);
    return out.join('');
  };

  const fence = (code) => `\`\`\`${L.fence}\n${code}\n\`\`\``;

  function paramsTable(key, params) {
    if (!params?.length) return '';
    const f = fill(key).params ?? {};
    const rows = params.map((p) => {
      const leaf = snake(p.name.replace(/^options\./, ''));
      const entry = f[leaf] ?? f[p.name];
      const given = typeof entry === 'object' && entry && !('python' in entry || 'nodejs' in entry) ? entry : { doc: entry };
      let doc = clean(p.doc, version) || say(given.doc, lang);
      if (!doc) unfilled.push(`${key}.${leaf}`);
      let def = p.default && p.default !== '...' ? `\`${p.default}\`` : given.default ? (given.default === 'none' ? 'none' : `\`${given.default}\``) : '';
      const optional = p.optional || (p.default != null);
      const name = `\`${p.name}\`${optional && !def && lang === 'nodejs' ? ' (optional)' : ''}`;
      return `| ${name} | ${cell(typeMd(p.type))} | ${cell(def || (optional ? '' : 'required'))} | ${cell(doc)} |`;
    });
    return `**Parameters**\n\n| Name | Type | Default | Description |\n| --- | --- | --- | --- |\n${rows.join('\n')}`;
  }

  function fieldsTable(name, fields) {
    if (!fields?.length) return '';
    const rows = fields.map((fl) => {
      const key = `${name}.${snake(fl.name)}`;
      const entry = FILL[key];
      const doc = clean(fl.doc, version) || say(typeof entry === 'object' && entry && 'doc' in entry ? entry.doc : entry, lang);
      if (!doc) unfilled.push(key);
      const def = typeof entry === 'object' && entry?.default ? ` Default \`${entry.default}\`.` : '';
      return `| \`${fl.name}\` | ${cell(typeMd(fl.type))} | ${fl.optional ? 'no' : 'yes'} | ${cell(doc + def)} |`;
    });
    return `**Fields**\n\n| Field | Type | Required | Description |\n| --- | --- | --- | --- |\n${rows.join('\n')}`;
  }

  function returnsMd(key, returns, isAsync) {
    const r = say(fill(key).returns, lang);
    if (!returns || returns === 'None' || returns === 'void' || returns === 'Promise<void>') return r ? `**Returns** ${r}` : '';
    if (returns === 'Self' || returns === 'this') return '**Returns** this builder, for chaining.';
    const t = isAsync ? `an awaitable of ${typeMd(returns)}` : typeMd(returns);
    return `**Returns** ${t}${r ? `: ${r}` : ''}`;
  }

  function raisesMd(key) {
    const r = say(fill(key).raises, lang);
    return r ? `**${RAISES[lang] === 'raises' ? 'Raises' : 'Throws'}** ${r}` : '';
  }

  // one function, method or property: heading, what it does, signature, parameters, returns
  function callable(key, m, level, display) {
    const blocks = [`${'#'.repeat(level)} \`${display}\``];
    const doc = docFor(key, m.doc);
    if (doc) blocks.push(doc);
    blocks.push(fence(m.signature));
    blocks.push(paramsTable(key, m.params));
    if (m.kind !== 'property' && m.kind !== 'constructor') blocks.push(returnsMd(key, m.returns, m.is_async));
    if (m.kind === 'property') blocks.push(`**Type** ${typeMd(m.type)}`);
    blocks.push(raisesMd(key));
    return blocks.filter(Boolean).join('\n\n');
  }

  const memberKey = (cls, m) => `${cls}.${m.kind === 'constructor' ? '__init__' : snake(m.name)}`;
  const memberTitle = (cls, m) => (m.kind === 'constructor' ? (lang === 'nodejs' ? `new ${cls}()` : `${cls}()`) : m.kind === 'property' ? m.name : `${m.name}()`);

  // a class as the page's subject: its members under Constructor, Properties, Methods
  function classPage(name, entry) {
    const groups = [
      ['Constructor', entry.members.filter((m) => m.kind === 'constructor')],
      ['Properties', entry.members.filter((m) => m.kind === 'property')],
      ['Methods', entry.members.filter((m) => m.kind === 'method')],
    ].filter(([, ms]) => ms.length);
    const glance = entry.members.map((m) => {
      const anchor = slug(memberTitle(name, m));
      return `| [\`${memberTitle(name, m)}\`](#${anchor}) | ${cell(firstSentence(docFor(memberKey(name, m), m.doc)))} |`;
    });
    return [
      `## Members\n\n| Member | Description |\n| --- | --- |\n${glance.join('\n')}`,
      ...groups.map(([title, ms]) => `## ${title}\n\n${ms.map((m) => callable(memberKey(name, m), m, 3, memberTitle(name, m))).join('\n\n')}`),
    ].join('\n\n');
  }

  // a name on a page of several: a function, a type, or a class-like type with methods
  function item(name, entry) {
    const key = canon(name);
    if (entry.kind === 'function') return callable(key, entry, 2, `${name}()`);
    const blocks = [`## \`${name}\``];
    const doc = docFor(key, entry.doc);
    if (doc) blocks.push(doc);
    if (entry.definition) blocks.push(fence(entry.definition));
    blocks.push(fieldsTable(name, entry.fields));
    for (const m of entry.members ?? []) blocks.push(callable(`${name}.${snake(m.name)}`, m, 3, `${m.name}()`));
    blocks.push(raisesMd(key));
    return blocks.filter(Boolean).join('\n\n');
  }

  function importLine(names) {
    const by = {};
    for (const n of names) (by[api[n].module] ??= []).push(n);
    if (lang === 'python') {
      return Object.entries(by)
        .map(([mod, ns]) => (mod === 'ailoy._ailoy' ? `if TYPE_CHECKING:\n    from ailoy._ailoy import ${ns.join(', ')}` : `from ${mod} import ${ns.join(', ')}`))
        .join('\n');
    }
    const values = names.filter((n) => !api[n].type_only);
    const types = names.filter((n) => api[n].type_only);
    return [
      values.length && `import { ${values.join(', ')} } from '@brekkylab/ailoy';`,
      types.length && `import type { ${types.join(', ')} } from '@brekkylab/ailoy';`,
    ].filter(Boolean).join('\n');
  }

  // `next: false` on the language's last page: the sidebar's next entry is the other language's
  // Introduction, which a reader of this language never meant to go to
  const frontmatter = (title, description, last = false) => `---
title: ${JSON.stringify(title)}
description: ${JSON.stringify(description)}${last ? '\nnext: false' : ''}
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---
`;

  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });

  const has = (page) => page.items.some((n) => api[own(n)]) || Boolean(page.intro);
  const bySlug = Object.fromEntries(PAGES.map((p) => [p.slug, p]));
  const last = SECTIONS.flatMap((s) => s.pages).filter((s) => s !== 'index' && bySlug[s] && has(bySlug[s])).at(-1);

  const written = [];
  for (const page of PAGES) {
    const names = page.items.map(own).filter((n) => api[n]);
    if (!has(page)) continue;
    const blocks = [frontmatter(page.title, page.summary, page.slug === last).trimEnd(), say(page.intro, lang)];
    const ex = example(page.slug, lang);
    if (ex) blocks.push(`## Example

\`\`\`${lang === 'python' ? 'python' : 'js'}
${ex}
\`\`\``);
    else if (names.length) blocks.push(fence(importLine(names)));
    const guides = page.guides.map((gd) => `[${gd.title}](/docs/${gd.slug}/)`).join(', ');
    blocks.push(`Guides: ${guides}.`);
    if (names.length === 1 && api[names[0]].kind === 'class' && api[names[0]].module !== 'ailoy._ailoy' && page.items.length === 1) {
      if (!page.intro) blocks.push(docFor(names[0], api[names[0]].doc));
      blocks.push(classPage(names[0], api[names[0]]));
    } else if (names.length) {
      const glance = names.map((n) => {
        const e = api[n];
        const title = e.kind === 'function' ? `${n}()` : n;
        const kind = e.kind === 'function' ? 'function' : e.kind === 'class' ? (e.module === 'ailoy._ailoy' || lang === 'nodejs' ? 'interface' : 'class') : 'type';
        const doc = docFor(canon(n), e.doc) || '';
        return `| [\`${title}\`](#${n.toLowerCase()}) | ${kind} | ${cell(firstSentence(doc))} |`;
      });
      blocks.push(`## On this page\n\n| Name | Kind | Description |\n| --- | --- | --- |\n${glance.join('\n')}`);
      blocks.push(...names.map((n) => item(n, api[n])));
    }
    fs.writeFileSync(path.join(outDir, `${page.slug}.md`), blocks.filter(Boolean).join('\n\n') + '\n');
    written.push(page.slug);
  }

  fs.writeFileSync(path.join(outDir, 'index.md'), overview(lang, version, tag, written));
  return { slugs: written, unfilled: [...new Set(unfilled)] };
}

function overview(lang, version, tag, written) {
  const L = LANGS[lang];
  const install = lang === 'python' ? 'pip install ailoy-py' : 'npm install @brekkylab/ailoy';
  const first = lang === 'python' ? 'from ailoy import AgentBuilder' : "const { AgentBuilder } = require('@brekkylab/ailoy')";
  const flow = lang === 'python'
    ? '[`AgentBuilder`](agent-builder/) configures an agent, and `await builder.build()` makes an [`Agent`](agent/). `agent.run(query)` returns an [`AgentRun`](agent-run/), the turn, which you iterate with `async for` to get each [`MessageOutput`](outputs/#messageoutput). `run_stream` gives [`MessageDeltaOutput`](outputs/#messagedeltaoutput)s as the model writes.'
    : '[`AgentBuilder`](agent-builder/) configures an agent, and `await builder.build()` makes an [`Agent`](agent/). `agent.run(query)` returns an [`AgentRun`](agent-run/), the turn, which you iterate with `for await` to get each [`MessageOutput`](outputs/#messageoutput). `runStream` gives [`MessageDeltaOutput`](outputs/#messagedeltaoutput)s as the model writes.';
  const bySlug = Object.fromEntries(PAGES.map((p) => [p.slug, p]));
  const sections = SECTIONS.map((s) => {
    const rows = s.pages.filter((p) => p !== 'index' && written.includes(p)).map((p) => `| [${bySlug[p].title}](${p}/) | ${bySlug[p].summary} |`);
    return rows.length ? `### ${s.title}\n\n| Page | What it covers |\n| --- | --- |\n${rows.join('\n')}` : '';
  }).filter(Boolean);
  return `---
title: ${JSON.stringify(`${L.label} installation`)}
description: ${JSON.stringify(`Install ${L.pkg}, import it, and see how its classes fit together.`)}
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

This page installs \`${L.pkg}\` and shows how its classes fit together. New here? Start with the [introduction](/docs/reference/).

## Install

\`\`\`sh
${install}
\`\`\`

${lang === 'python' ? 'It needs Python 3.10 or later. Every class and function is imported from `ailoy`, and the dict shapes from `ailoy.types`:' : 'It needs Node.js 18 or later. Every class, function and type is imported from `@brekkylab/ailoy`:'}

\`\`\`${lang === 'python' ? 'python' : 'js'}
${first}
\`\`\`

## How the pieces fit

${flow}

Models and tools are set up process-wide, before an agent is built: [Model providers](model-providers/) decide how each model runs, and [Tools and MCP](tools/) register what runs behind each tool name. The [data shapes](messages/) are plain ${lang === 'python' ? 'dicts' : 'objects'} in their JSON form.

## Pages

${sections.join('\n\n')}
`;
}
