import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

// The docs as ailoy.co/docs serves them: same Starlight setup, styles and header.
// Keep this in step with ailoy.co's astro.config.mjs. The differences: the site is served under
// base /docs/ (ailoy.co has src/content/docs/docs/ instead, since its root is the product site),
// so the sidebar slugs have no docs/ prefix; the header's Ailoy name is not a link (the site root
// lives on ailoy.co); the sidebar is the one of the sandbox release, which ailoy.co gets with
// that release, and has no Tutorials or Examples entry.

const FONTS =
  'https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap';

// With base /docs/ the site root is outside the site; `npm run dev` sends it to /docs/.
const rootToDocs = {
  name: 'root-to-docs',
  // Astro puts its base handling, which answers / with a 404, at the front of the stack once the
  // server is configured; this runs after it (post) and goes in front of it.
  enforce: 'post',
  configureServer(server) {
    return () => server.middlewares.stack.unshift({ route: '', handle: redirectRoot });
  }
};
function redirectRoot(req, res, next) {
  if (req.url !== '/') return next();
  res.writeHead(302, { Location: '/docs/' });
  res.end();
}

export default defineConfig({
  site: 'https://ailoy.co',
  trailingSlash: 'always',
  // Building VM became Giving an agent a sandbox.
  redirects: { '/guide/building-vm/': '/docs/guide/sandbox/' },
  build: { format: 'directory' },
  base: '/docs',
  integrations: [
    starlight({
      title: 'Ailoy Docs',
      description: 'Documentation for Ailoy, the open-source agent SDK built around a sandbox.',
      favicon: '/favicon.ico',
      head: [
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
        { tag: 'link', attrs: { rel: 'stylesheet', href: FONTS } }
      ],
      customCss: ['./src/styles/starlight.css'],
      components: {
        SiteTitle: './src/components/docs/SiteTitle.astro',
        ThemeSelect: './src/components/docs/ThemeSelect.astro',
        ThemeProvider: './src/components/docs/ThemeProvider.astro'
      },
      // Fails the build on a broken absolute link or heading anchor, such as a path without
      // /docs/, which would otherwise build and break only once served. Relative links
      // (../tools/) are allowed and not checked: they resolve the same here and on ailoy.co.
      plugins: [starlightLinksValidator({ errorOnRelativeLinks: false })],
      // Plain code frames everywhere: shell blocks lose the terminal window dots.
      expressiveCode: { defaultProps: { frame: 'code' } },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/brekkylab/ailoy' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/27rx3EJy3P' }
      ],
      sidebar: [
        {
          label: 'Getting started',
          items: [
            { label: 'Overview', link: '/' },
            { slug: 'guide/installation' },
            { slug: 'guide/quick-start' }
          ]
        },
        { slug: 'concepts' },
        {
          label: 'Building an agent',
          items: [
            { slug: 'guide/agents' },
            { slug: 'guide/sandbox' },
            { slug: 'guide/running' },
            { slug: 'guide/model-providers' },
            { slug: 'guide/message-format' }
          ]
        },
        {
          label: 'Tools',
          items: [
            { slug: 'guide/tools' },
            { slug: 'guide/using-mcp' },
            { slug: 'guide/using-browser' },
            { slug: 'guide/skills' }
          ]
        },
        {
          label: 'State over time',
          items: [{ slug: 'guide/context-manager' }]
        },
        { slug: 'reference' }
      ]
    })
  ],
  vite: { plugins: [rootToDocs] }
});
