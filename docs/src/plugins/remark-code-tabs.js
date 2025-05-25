const { visit } = require("unist-util-visit");

module.exports = function remarkCodeTabs() {
  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (node.type !== "mdxJsxFlowElement" || node.name !== "CodeTabs") return;

      const codeNodes = node.children.filter((c) => c.type === "code");

      const tabItems = codeNodes.map((codeNode) => {
        const lang = codeNode.lang;

        // Determine TabItem value and label
        let tabValue;
        let tabLabel;
        if (lang === "typescript" || lang === "javascript") {
          // TODO: Distinguish with Javascript(Web) after Emscripten is supported
          tabValue = "node";
          tabLabel = "JavaScript(Node)";
        } else {
          tabValue = lang;
          tabLabel = lang.charAt(0).toUpperCase() + lang.slice(1);
        }

        // Add "showLineNumbers" in code meta
        let meta = codeNode.meta;
        if (meta === null) {
          meta = "showLineNumbers";
        } else if (!meta.includes("showLineNumbers")) {
          meta += " showLineNumbers";
        }

        return {
          type: "mdxJsxFlowElement",
          name: "TabItem",
          attributes: [
            { type: "mdxJsxAttribute", name: "value", value: tabValue },
            { type: "mdxJsxAttribute", name: "label", value: tabLabel },
          ],
          children: [
            {
              ...codeNode,
              meta,
            },
          ],
        };
      });

      const codeTabsNode = {
        type: "mdxJsxFlowElement",
        name: "Tabs",
        attributes: [
          {
            type: "mdxJsxAttribute",
            name: "groupId",
            value: "code-language",
          },
        ],
        children: tabItems,
      };

      parent.children[index] = codeTabsNode;
    });
  };
};
