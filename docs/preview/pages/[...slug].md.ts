// Every page as plain Markdown at its address plus .md (/docs/guides/tools.md), for Copy Markdown,
// View as Markdown and AI tools. The page's source with its components turned into Markdown: each
// language tab becomes a bold label over its code, each link card a list item, and the diagram,
// which has no text form, is left out.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';

export const getStaticPaths: GetStaticPaths = async () =>
  (await getCollection('docs')).map((entry) => ({ params: { slug: entry.id }, props: { entry } }));

export const GET: APIRoute = ({ props, site }) => {
  const { data, body = '' } = props.entry;
  const text = body
    .replace(/^import .*\n/gm, '')
    .replace(/^[ \t]*<\/?(Tabs|CardGrid)\b[^>]*>[ \t]*(\n|$)/gm, '')
    .replace(/^[ \t]*<\/TabItem>[ \t]*(\n|$)/gm, '')
    .replace(/^[ \t]*<TabItem\b[^>]*\blabel="([^"]+)"[^>]*>[ \t]*$/gm, '**$1**')
    .replace(/^[ \t]*<LinkCard\b([^>]*)\/>[ \t]*$/gm, (_, attrs: string) => {
      const a = (k: string) => attrs.match(new RegExp(`\\b${k}="([^"]*)"`))?.[1];
      return `- [${a('title')}](${a('href')})${a('description') ? `: ${a('description')}` : ''}`;
    })
    .replace(/^[ \t]*<HowDiagram\b[^>]*\/>[ \t]*(\n|$)/gm, '')
    .replace(/^[ \t]*<\/?div\b[^>]*>[ \t]*(\n|$)/gm, '')
    // Full addresses, so links still work once the page is pasted into a chat.
    .replace(/\]\(\/docs\//g, `](${new URL('/docs/', site)}`)
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const head = `# ${data.title}\n\n${data.description ? data.description + '\n\n' : ''}`;
  return new Response(head + text + '\n', { headers: { 'Content-Type': 'text/markdown; charset=utf-8' } });
};
