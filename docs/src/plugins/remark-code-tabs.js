const { visit } = require("unist-util-visit");

module.exports = function remarkCodeTabs() {
  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (node.type !== "mdxJsxFlowElement" || node.name !== "CodeTabs") return;

      const codeNodes = node.children.filter((c) => c.type === "code");

      const tabItems = codeNodes.map((codeNode) => {
        // Add "showLineNumbers" in code meta
        let meta = codeNode.meta;
        if (meta === null) {
          meta = "showLineNumbers";
        } else if (!meta.includes("showLineNumbers")) {
          meta += " showLineNumbers";
        }

        // Determine TabItem value and label
        const lang = codeNode.lang;
        let tabValue;
        let tabLabel;
        if (lang === "typescript" || lang === "javascript") {
          if (meta.includes("web")) {
            tabValue = "web";
            tabLabel = "JavaScript(Web)";
          } else if (meta.includes("node")) {
            tabValue = "node";
            tabLabel = "JavaScript(Node)";
          } else {
            tabValue = "javascript";
            tabLabel = "JavaScript";
          }
        } else {
          tabValue = lang;
          tabLabel = lang.charAt(0).toUpperCase() + lang.slice(1);
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
