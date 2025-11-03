import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  documentSidebar: [
    {
      type: "category",
      label: "Tutorial",
      items: [
        "tutorial/getting-started",
        "tutorial/system-message",
        "tutorial/multi-turn-conversation",
        "tutorial/reasoning",
        "tutorial/using-tools",
        "tutorial/mcp-integration",
        "tutorial/rag-using-documents",
        "tutorial/managing-model-files",
        "tutorial/webassembly-supports",
      ],
    },
    {
      type: "category",
      label: "Concepts",
      items: [
        "concepts/how-agent-works",
        "concepts/architecture",
        "concepts/chat-completion-format",
      ],
    },
    {
      type: "category",
      label: "Resources",
      items: [
        "resources/available-models",
        "resources/supported-environments",
        "resources/command-line-interfaces",
      ],
    },
    {
      type: "category",
      label: "API References",
      items: [
        {
          type: "html",
          value: `<a href="/ailoy/api-references/python/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">Python API References</a>`,
          defaultStyle: true,
        },
        {
          type: "html",
          value: `<a href="/ailoy/api-references/nodejs/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">JavaScript(Node.js) API References</a>`,
          defaultStyle: true,
        },
        {
          type: "html",
          value: `<a href="/ailoy/api-references/js-web/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">JavaScript(Web) API References</a>`,
          defaultStyle: true,
        },
      ],
    },
  ],
};

export default sidebars;
