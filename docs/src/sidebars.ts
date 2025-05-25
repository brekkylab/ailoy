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
        "tutorial/multi-turn-conversation",
        "tutorial/reasoning",
        "tutorial/using-tools",
        "tutorial/integrate-with-mcp",
        "tutorial/calling-low-level-apis",
        "tutorial/rag-with-vector-store",
      ],
    },
    {
      type: "category",
      label: "Concepts",
      items: [
        "concepts/architecture",
        "concepts/agent-response-format",
        "concepts/tools",
        "concepts/command-line-interfaces",
      ],
    },
    {
      type: "category",
      label: "API References",
      items: [
        "api-references/internal-runtime-apis",
        {
          type: "html",
          value: `<a href="/pydocs/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">Python API References</a>`,
          defaultStyle: true,
        },
        {
          type: "html",
          value: `<a href="/tsdocs/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">Javascript(Node) API References</a>`,
          defaultStyle: true,
        },
      ],
    },
  ],
};

export default sidebars;
