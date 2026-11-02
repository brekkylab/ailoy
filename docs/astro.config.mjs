import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The docs as ailoy.co/docs serves them: same Starlight setup, styles and header.
// Keep this in step with ailoy.co's astro.config.mjs. The only differences: the root redirects
// to /docs/, the header's Ailoy name is not a link (the site root lives on ailoy.co), and there
// is no Examples entry.

const FONTS =
  'https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap';

export default defineConfig({
  site: 'https://ailoy.co',
  trailingSlash: 'always',
  build: { format: 'directory' },
  // The docs live under /docs/, as on ailoy.co; the root only sends you there, so the address
  // `npm run dev` prints opens the docs.
  redirects: { '/': '/docs/' },
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
            { label: 'Overview', link: '/docs/' },
            { slug: 'docs/guide/quick-start' }
          ]
        },
        {
          label: 'Guides',
          items: [
            { slug: 'docs/guide/model-providers' },
            { slug: 'docs/guide/building-vm' },
            { slug: 'docs/guide/message-format' },
            { slug: 'docs/guide/tools' },
            { slug: 'docs/guide/skills' },
            { slug: 'docs/guide/context-manager' },
            { slug: 'docs/guide/using-mcp' },
            { slug: 'docs/guide/using-browser' }
          ]
        },
        // one page each for now: a plain link, not a group holding one item of the same name
        { slug: 'docs/concepts' },
        { slug: 'docs/reference' }
      ]
    })
  ]
});
