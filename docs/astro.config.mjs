import { readdirSync, readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// A preview of the docs, with ailoy.co's styles and header. ailoy.co serves them at
// ailoy.co/docs/ from this same folder: what both read is content/ (the pages) and sidebar.json
// (the sidebar's groups and order); see README.md for what a page may use.

const sidebar = JSON.parse(readFileSync(new URL('./sidebar.json', import.meta.url), 'utf8'));

// The guides were under guide/ before they moved to guides/; the old addresses still lead there.
// Building VM became Giving an agent a sandbox.
const guides = readdirSync(new URL('./content/guides/', import.meta.url)).map((f) => f.replace(/\.mdx?$/, ''));
const redirects = Object.fromEntries([
  ...guides.map((g) => [`/guide/${g}/`, `/docs/guides/${g}/`]),
  ['/guide/building-vm/', '/docs/guides/sandbox/']
]);

const FONTS =
  'https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap';

// A code theme with one colour for code and a dimmer one for comments.
const mono = (type, bg, fg, comment) => ({
  name: `ailoy-${type}`,
  type,
  colors: { 'editor.background': bg, 'editor.foreground': fg },
  tokenColors: [
    { settings: { foreground: fg } },
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: comment, fontStyle: 'italic' } }
  ]
});

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
  redirects,
  build: { format: 'directory' },
  base: '/docs',
  // the preview's own code (content config, components, styles) is kept apart from the pages
  srcDir: './preview',
  integrations: [
    starlight({
      title: 'Ailoy Docs',
      description: 'Documentation for Ailoy, the open-source agent SDK built around a sandbox.',
      favicon: '/favicon.ico',
      head: [
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
        { tag: 'link', attrs: { rel: 'stylesheet', href: FONTS } }
      ],
      customCss: ['./preview/styles/starlight.css'],
      editLink: { baseUrl: 'https://github.com/brekkylab/ailoy/edit/main/docs/' },
      components: {
        Header: './preview/components/docs/Header.astro',
        Sidebar: './preview/components/docs/Sidebar.astro',
        PageTitle: './preview/components/docs/PageTitle.astro',
        Footer: './preview/components/docs/Footer.astro',
        SiteTitle: './preview/components/docs/SiteTitle.astro',
        ThemeSelect: './preview/components/docs/ThemeSelect.astro',
        ThemeProvider: './preview/components/docs/ThemeProvider.astro'
      },
      // Plain code frames everywhere: shell blocks lose the terminal window dots.
      // Monochrome code in square hairline frames, as on ailoy.co: one ink, comments dimmer, and
      // yellow only for the lines a page marks.
      expressiveCode: {
        themes: [mono('dark', '#0e0f11', '#d9d8d3', '#6c7079'), mono('light', '#f3f3f1', '#24262b', '#80838b')],
        defaultProps: { frame: 'code' },
        styleOverrides: {
          borderRadius: '0',
          borderColor: 'var(--sl-color-hairline)',
          codeBackground: 'var(--sl-color-gray-6)',
          frames: { frameBoxShadowCssValue: 'none', editorBackground: 'var(--sl-color-gray-6)' },
          textMarkers: {
            markBackground: 'rgba(255, 197, 61, 0.10)',
            markBorderColor: '#ffc53d',
            insBackground: 'rgba(255, 197, 61, 0.06)',
            insBorderColor: 'rgba(255, 197, 61, 0.5)',
            insDiffIndicatorColor: '#ffc53d'
          }
        }
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/brekkylab/ailoy' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/27rx3EJy3P' }
      ],
      sidebar
    })
  ],
  vite: { plugins: [rootToDocs] }
});
