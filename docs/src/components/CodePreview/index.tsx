import { python } from "@codemirror/lang-python";
import {
  SandpackCodeEditor,
  SandpackFileExplorer,
  SandpackLayout,
  SandpackProvider,
} from "@codesandbox/sandpack-react";
import { amethyst } from "@codesandbox/sandpack-themes";
import { useMediaQuery } from "@site/src/hooks/use-media-query";
import CodeBlock from "@theme/CodeBlock";
import clsx from "clsx";
import { useState } from "react";
import styles from "./style.module.scss";

type SupportedLanguage = "python" | "nodejs" | "web";

const pythonFiles = {
  "main.py": {
    active: true,
    code: `from ailoy import Runtime, Agent, LocalModel

# Start Ailoy runtime
rt = Runtime()

# Create an agent
# During this step, the model parameters are downloaded and the LLM is set up for execution
agent = Agent(rt, LocalModel("Qwen/Qwen3-0.6B"))

# Agent answers within the agentic loop
for resp in agent.query("Please give me a short poem about AI"):
    agent.print(resp)

# Once the agent is no longer needed, it can be released
agent.delete()

# Stop the runtime
rt.stop()`,
  },
  "pyproject.toml": {
    code: `[project]
name = "ailoy-python-example"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ailoy-py>=0.1.0",
]
`,
  },
};

const nodejsFiles = {
  "index.js": {
    active: true,
    code: `import * as ai from "ailoy-node";

(async () => {
  // Start Ailoy runtime
  const rt = await ai.startRuntime();

  // Create an agent
  // During this step, the model parameters are downloaded and the LLM is set up for execution
  const agent = await ai.defineAgent(
    rt,
    ai.LocalModel({ id: "Qwen/Qwen3-0.6B" })
  );

  // Agent answers within the agentic loop
  for await (const resp of agent.query(
    "Please give me a short poem about AI"
  )) {
    agent.print(resp);
  }

  // Once the agent is no longer needed, it can be released
  await agent.delete();

  // Stop the runtime
  await rt.stop();
})();
`,
  },
  "package.json": {
    code: `{
  "name": "ailoy-nodejs-example",
  "version": "0.1.0",
  "type": "module",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "ailoy-node": "^0.1.0"
  }
}
`,
  },
};

const webFiles = {
  "index.js": {
    active: true,
    code: `import * as ai from "ailoy-web";

// Start Ailoy runtime
const rt = await ai.startRuntime();

// Create an agent
// During this step, the model parameters are downloaded and the LLM is set up for execution
const agent = await ai.defineAgent(
  rt,
  ai.LocalModel({ id: "Qwen/Qwen3-0.6B" })
);

document.getElementById("submit").addEventListener("click", async () => {
  const query = document.getElementById("query").value;
  const textarea = document.getElementById("answer");
  textarea.innerHTML = "";

  // Agent answers within the agentic loop
  for await (const resp of agent.query(query)) {
    textarea.innerHTML += resp.content;
  }
});
`,
  },
  "index.html": {
    code: `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  </head>
  <body>
    <script type="module" src="/index.js"></script>
    <div style="display: flex;">
        <input type="text" id="query"></input>
        <button id="submit">Submit</button>
    </div>
    <textarea id="answer" rows="10"></textarea>
  </body>
</html>
`,
  },
  "package.json": {
    code: `{
  "name": "ailoy-web-example",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite"
  },
  "dependencies": {
    "ailoy-web": "^0.1.0"
  },
  "devDependencies": {
    "vite": "^7.1.2"
  }
}`,
  },
  "vite.config.js": {
    code: `import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["ailoy-web"],
  },
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ailoy: ["ailoy-web"],
        },
      },
    },
  },
});
`,
  },
};

const sandpackProviderProps = {
  python: {
    files: pythonFiles,
  },
  nodejs: {
    files: nodejsFiles,
  },
  web: {
    files: webFiles,
  },
};

const installCommands = {
  python: `pip install ailoy-py`,
  nodejs: `npm install ailoy-node`,
  web: `npm install ailoy-web`,
};

export default function CodePreview() {
  const [activeLang, setActiveLang] = useState<SupportedLanguage>("python");
  const isMobile = useMediaQuery("(max-width: 768px)");

  return (
    <section>
      <div className="container">
        {/* Language Tabs*/}
        <div className="tabs-container">
          <ul role="tablist" aria-orientation="horizontal" className="tabs">
            <li
              role="tab"
              tabIndex={0}
              className={clsx(
                "tabs__item",
                activeLang === "python" && "tabs__item--active"
              )}
              onClick={() => setActiveLang("python")}
            >
              Python
            </li>
            <li
              role="tab"
              tabIndex={0}
              className={clsx(
                "tabs__item",
                activeLang === "nodejs" && "tabs__item--active"
              )}
              onClick={() => setActiveLang("nodejs")}
            >
              Javascript(Node.js)
            </li>
            <li
              role="tab"
              tabIndex={0}
              className={clsx(
                "tabs__item",
                activeLang === "web" && "tabs__item--active"
              )}
              onClick={() => setActiveLang("web")}
            >
              Javascript(Web)
            </li>
          </ul>
        </div>

        <CodeBlock
          language="bash"
          className={styles["codeblock--install-command"]}
        >{`$ ${installCommands[activeLang]}`}</CodeBlock>

        {/* Code Viewer */}
        <SandpackProvider
          {...sandpackProviderProps[activeLang]}
          theme={{
            ...amethyst,
            colors: {
              ...amethyst.colors,
              accent: "#7fc4d9",
            },
            font: {
              size: isMobile ? "9px" : "12px",
            },
          }}
          style={{
            height: "50vh",
            border: "2px solid #282c34",
            borderRadius: "0 0 6px 6px",
            fontSize: isMobile ? "9px" : "12px",
            backgroundColor: "#282c34",
          }}
        >
          <SandpackLayout
            style={{
              border: 0,
              height: "100%",
            }}
          >
            {!isMobile && (
              <SandpackFileExplorer
                className={styles[`sandpack--file-explorer--${activeLang}`]}
                style={{ height: "auto" }}
              />
            )}
            <SandpackCodeEditor
              wrapContent
              readOnly
              showTabs
              additionalLanguages={[
                {
                  name: "python",
                  extensions: ["py"],
                  language: python(),
                },
              ]}
              style={{ height: "100%", fontSize: isMobile ? "9px" : "12px" }}
            ></SandpackCodeEditor>
          </SandpackLayout>
        </SandpackProvider>
      </div>
    </section>
  );
}
