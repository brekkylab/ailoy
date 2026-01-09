import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes as prismThemes } from "prism-react-renderer";
import remarkCodeTabs from "./src/plugins/remark-code-tabs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "Build your AI Agents instantly",
  tagline: "in any environments, in any languages, without any complex setup.",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://brekkylab.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/ailoy/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "brekkylab", // Usually your GitHub org/user name.
  projectName: "ailoy", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Internationalization configuration
  i18n: {
    defaultLocale: "en",
    locales: ["en", "ko"],
    localeConfigs: {
      en: { label: "English" },
      ko: { label: "한국어" },
    },
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./src/sidebars.ts",
          remarkPlugins: [remarkCodeTabs],
        },
        theme: {
          customCss: ["./src/css/custom.css", "./src/css/navbar.css"],
        },
      } satisfies Preset.Options,
    ],
  ],

  markdown: {
    mermaid: true,
  },

  plugins: ["docusaurus-plugin-sass"],

  themes: ["@docusaurus/theme-mermaid"],

  themeConfig: {
    // Replace with your project's social card
    // image: "img/docusaurus-social-card.jpg",
    colorMode: {
      disableSwitch: true,
      defaultMode: "dark",
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: "Ailoy",
      style: "dark",
      logo: {
        alt: "Ailoy Logo",
        src: "img/logo.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "documentSidebar",
          position: "left",
          label: "Documents",
        },
        {
          href: "https://medium.com/ailoy",
          label: "Blog",
          position: "left",
        },
        {
          href: "https://github.com/brekkylab/ailoy",
          position: "right",
          className: "navbar--github-link",
        },
        {
          href: "https://discord.gg/27rx3EJy3P",
          position: "right",
          className: "navbar--discord-link",
        },
        {
          type: "localeDropdown",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          items: [
            {
              html: `<img src="/ailoy/img/ailoy-logo-letter.png" width="200px">`,
            },
            {
              html: `<a class="footer__link-item" href="mailto:contact@brekkylab.com">contact@brekkylab.com</a>`,
            },
            {
              html: `<p>Copyright © ${new Date().getFullYear()} Brekkylab, Inc.</p>`,
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "Getting Started",
              to: "/docs/tutorial/getting-started",
            },
            {
              html: `<a class="footer__link-item" href="/ailoy/docs/api-references/python/" target="_blank" rel="noopener noreferrer">Python API References</a>`,
            },
            {
              html: `<a class="footer__link-item" href="/ailoy/docs/api-references/nodejs/" target="_blank" rel="noopener noreferrer">Javascript(Node.js) API References</a>`,
            },
            {
              html: `<a class="footer__link-item" href="/ailoy/docs/api-references/js-web/" target="_blank" rel="noopener noreferrer">Javascript(Web) API References</a>`,
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/brekkylab/ailoy",
            },
            {
              label: "Discord",
              href: "https://discord.gg/27rx3EJy3P",
            },
            {
              label: "X",
              href: "https://x.com/ailoy_co",
            },
            {
              label: "Linkedin",
              href: "https://www.linkedin.com/company/107147231",
            },
            {
              label: "Blog",
              href: "https://medium.com/ailoy",
            },
          ],
        },
      ],
      // copyright: `Copyright © ${new Date().getFullYear()} Brekkylab, Inc.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: {
        plain: {
          color: "#e2e8f0",
          backgroundColor: "#282c34",
        },
        styles: [
          {
            types: ["comment", "prolog", "doctype", "cdata"],
            style: {
              color: "#8b949e", // Light gray - comments
              fontStyle: "italic",
            },
          },
          {
            types: ["namespace"],
            style: {
              opacity: 0.7,
            },
          },
          {
            types: ["string", "attr-value", "char", "regex"],
            style: {
              color: "#a5d6ff", // Light blue (Strings)
            },
          },
          {
            types: ["punctuation", "operator"],
            style: {
              color: "#c9d1d9", // Light gray (Punctuation marks, operators)
            },
          },
          {
            types: [
              "entity",
              "url",
              "symbol",
              "number",
              "boolean",
              "variable",
              "constant",
              "property",
              "inserted",
            ],
            style: {
              color: "#79c0ff", // Light blue (Variables, numbers, booleans)
            },
          },
          {
            types: ["keyword", "selector"],
            style: {
              color: "#ff7b72", // Reddish (Keywords, like `import`, `const`, `async`)
            },
          },
          {
            types: ["atrule", "attr-name"],
            style: {
              color: "#d2a8ff", // Light purple (CSS @rules and HTML attribute names)
            },
          },
          {
            types: ["function", "deleted", "tag"],
            style: {
              color: "#d2a8ff", // Light purple (Functions, JSX/HTML tags)
            },
          },
          {
            types: ["function-variable"],
            style: {
              color: "#f778ba", // Pink (Function variables)
            },
          },
          {
            types: ["class-name", "builtin"],
            style: {
              color: "#ffa657", // Orange (Class names, like `React.FC`)
            },
          },
        ],
      },

      additionalLanguages: ["bash"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
