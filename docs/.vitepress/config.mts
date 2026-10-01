import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Ailoy',
  description: 'Give your agent a computer of its own.',
  // Served from https://brekkylab.github.io/ailoy/.
  base: '/ailoy/',
  cleanUrls: true,
  lastUpdated: true,
  head: [['link', { rel: 'icon', href: '/ailoy/img/favicon.ico' }]],

  themeConfig: {
    logo: '/img/ailoy-logo-letter.png',
    siteTitle: false,

    nav: [
      { text: 'Quick start', link: '/guide/quick-start' },
      { text: 'Examples', link: '/guide/examples' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Quick start', link: '/guide/quick-start' },
            { text: 'Examples', link: '/guide/examples' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/brekkylab/ailoy' },
      { icon: 'discord', link: 'https://discord.gg/27rx3EJy3P' },
      { icon: 'x', link: 'https://x.com/ailoy_co' },
    ],

    editLink: {
      pattern: 'https://github.com/brekkylab/ailoy/edit/main/docs/:path',
    },

    search: { provider: 'local' },

    footer: {
      message: 'Released under the Apache-2.0 License.',
      copyright: '© Brekkylab',
    },
  },
})
