import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The docs as ailoy.co/docs serves them: same Starlight setup, styles and header.
// Keep this in step with ailoy.co's astro.config.mjs. The differences: the site is served under
// base /docs/ (ailoy.co has src/content/docs/docs/ instead, since its root is the product site),
// so the sidebar slugs have no docs/ prefix; the header's Ailoy name is not a link (the site root
// lives on ailoy.co); and there is no Examples entry.

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
            { slug: 'guide/quick-start' }
          ]
        },
        {
          label: 'Guides',
          items: [
            { slug: 'guide/model-providers' },
            { slug: 'guide/building-vm' },
            { slug: 'guide/message-format' },
            { slug: 'guide/tools' },
            { slug: 'guide/skills' },
            { slug: 'guide/context-manager' },
            { slug: 'guide/using-mcp' },
            { slug: 'guide/using-browser' }
          ]
        },
        // one page each for now: a plain link, not a group holding one item of the same name
        { slug: 'concepts' },
        { slug: 'reference' }
      ]
    })
  ],
  vite: { plugins: [rootToDocs] }
});
