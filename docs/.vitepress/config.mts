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

    nav: [{ text: 'Guide', link: '/guide/quick-start', activeMatch: '/guide/' }],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Quick start', link: '/guide/quick-start' },
            { text: 'Message format', link: '/guide/message-format' },
            { text: 'Model providers', link: '/guide/model-providers' },
            { text: 'Building VM', link: '/guide/building-vm' },
            { text: 'Using MCP', link: '/guide/using-mcp' },
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
      copyright: `Copyright © ${new Date().getFullYear()} Brekkylab Inc.`,
    },
  },
})
