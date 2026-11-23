// Fits typedoc-plugin-markdown's output to the docs: kebab-case page names, links as /docs/ paths,
// Starlight frontmatter (noindex) and the line naming the release. regen.mjs sets the env.
import fs from 'node:fs';
import path from 'node:path';
import { Converter, ReflectionKind } from 'typedoc';
import { MarkdownPageEvent } from 'typedoc-plugin-markdown';

const TAG = process.env.AILOY_REF_TAG;
const SIDEBAR_OUT = process.env.AILOY_REF_SIDEBAR_OUT;
const VERSION = TAG?.replace(/^v/, '');
const BASE = '/docs/reference/nodejs/';
// napi's aliases of each class and its loader detail, not API
const HIDDEN = /^(Js[A-Z]\w*|__napiBindingTarget)$/;

const kebab = (s) => s.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
// "Class.AgentBuilder.md" -> "agent-builder"
const slugOf = (file) => kebab(path.basename(file, '.md').split('.').pop());

function summary(model) {
  const parts = model?.comment?.summary ?? model?.signatures?.[0]?.comment?.summary ?? [];
  const text = parts.map((p) => p.text).join('').replace(/\s+/g, ' ').trim();
  return text.split(/(?<=\.)\s/)[0].replace(/`/g, '').replace(/\[([^\]]+)\](?!\()/g, '$1');
}

export function load(app) {
  if (!TAG) throw new Error('AILOY_REF_TAG is not set: run node regen.mjs <tag>');

  app.converter.on(Converter.EVENT_RESOLVE_BEGIN, (context) => {
    for (const r of context.project.getReflectionsByKind(ReflectionKind.All)) {
      if (HIDDEN.test(r.name)) context.project.removeReflection(r);
    }
  });

  // the sidebar lists the classes and functions; the types are linked from the overview and
  // from every signature
  const sidebar = { classes: [], functions: [] };

  app.renderer.on(MarkdownPageEvent.BEGIN, (page) => {
    const name = page.model?.name ?? 'Node.js';
    const project = page.model?.kind === ReflectionKind.Project;
    if (!project) page.filename = path.join(path.dirname(page.filename), `${slugOf(page.filename)}.md`);
    const slug = `reference/nodejs/${path.basename(page.filename, '.md')}`;
    if (page.model?.kind === ReflectionKind.Class) sidebar.classes.push(slug);
    if (page.model?.kind === ReflectionKind.Function) sidebar.functions.push(slug);
    page.frontmatter = {
      title: project ? 'Node.js API reference' : name,
      description: project
        ? `The Node.js API of @brekkylab/ailoy ${VERSION}, generated from ${TAG}.`
        : summary(page.model) || `Node.js reference for ${name}.`,
      head: [{ tag: 'meta', attrs: { name: 'robots', content: 'noindex' } }],
      ...page.frontmatter,
    };
  });

  app.renderer.on(MarkdownPageEvent.END, (page) => {
    const line = `*Generated from the [${TAG}](https://github.com/brekkylab/ailoy/tree/${TAG}) release of \`@brekkylab/ailoy\`.*\n\n`;
    page.contents = page.contents
      // links between pages -> site paths
      .replace(/\]\(((?:[A-Za-z]+\.)?[\w$]+\.md)(#[^)]*)?\)/g, (_, f, h = '') =>
        `](${f === 'index.md' ? BASE : `${BASE}${slugOf(f)}/`}${h})`)
      // rustdoc links: to a crate module -> that module on docs.rs; any other [`X`] -> `X`
      .replace(/\[`crate::([\w:]+)`\](?!\()/g, (_, p) =>
        `[\`crate::${p}\`](https://docs.rs/ailoy/${VERSION}/ailoy/${p.replace(/::/g, '/')}/index.html)`)
      .replace(/\[(`[^`\]]+`)\](?!\()/g, '$1')
      // the docs use no em dashes; the doc comments use them as commas
      .replace(/\s*\u2014\s*/g, ', ')
      .replace(/^(---\n[\s\S]*?\n---\n\n?)/, `$1${line}`);
  });

  app.renderer.on('endRender', () => {
    if (SIDEBAR_OUT) fs.writeFileSync(SIDEBAR_OUT, JSON.stringify(sidebar));
  });
}
