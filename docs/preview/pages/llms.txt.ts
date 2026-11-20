// /docs/llms.txt: an index of the docs for AI tools (llmstxt.org), each page's title and
// description with a link to its Markdown copy, in the sidebar's order.
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import sidebar from '../../sidebar.json';

export const GET: APIRoute = async ({ site }) => {
  const pages = new Map((await getCollection('docs')).map((e) => [e.id === 'index' ? '' : e.id, e.data]));
  const md = (id: string) => new URL(`/docs/${id || 'index'}.md`, site).href;
  const line = (id: string) => {
    const d = pages.get(id);
    return d ? `- [${d.title}](${md(id)})${d.description ? `: ${d.description}` : ''}` : '';
  };
  const groups = sidebar.map(
    (g: any) =>
      `## ${g.label}\n\n` +
      g.items.map((i: any) => line(i.slug ?? (i.link === '/' ? '' : i.link))).filter(Boolean).join('\n')
  );
  const text = `# Ailoy\n\n> Ailoy is an open-source SDK for building AI agents in Python, Node.js or Rust, each with a Linux microVM of its own.\n\n${groups.join('\n\n')}\n`;
  return new Response(text, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
