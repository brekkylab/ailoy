// Every page as plain Markdown at its address plus .md (/docs/guides/tools.md), for Copy Markdown,
// View as Markdown and AI tools. The MDX imports are dropped; the rest is the page's source.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';

export const getStaticPaths: GetStaticPaths = async () =>
  (await getCollection('docs')).map((entry) => ({ params: { slug: entry.id }, props: { entry } }));

export const GET: APIRoute = ({ props }) => {
  const { data, body = '' } = props.entry;
  const text = body.replace(/^import .*\n/gm, '').trim();
  const head = `# ${data.title}\n\n${data.description ? data.description + '\n\n' : ''}`;
  return new Response(head + text + '\n', { headers: { 'Content-Type': 'text/markdown; charset=utf-8' } });
};
