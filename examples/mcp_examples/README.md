# Ailoy MCP Integration Examples

These examples show how to integrate various MCP (Model Context Protocol) servers with the **Ailoy** agent.

## GitHub MCP Server

This example demonstrates how to use the [GitHub MCP Server](https://github.com/github/github-mcp-server) to interact with GitHub through the Ailoy agent. You can instruct the agent to perform a variety of GitHub-related tasks, including:
* Searching for repositories
* Fetching file contents
* Creating issues or pull requests
* And more

### Prerequisites

As outlined in the [GitHub MCP Server documentation](https://github.com/github/github-mcp-server?tab=readme-ov-file#prerequisites), you'll need the following:

* [Docker](https://www.docker.com/)
* A [GitHub Personal Access Token](https://github.com/settings/personal-access-tokens/new)

> ⚠️ The GitHub MCP server requires the `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable, even if you don't plan to use tools that require authentication.


### How to Run

```bash
# Set the `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable
export GITHUB_PERSONAL_ACCESS_TOKEN=xxx
uv run github.py
```

---

## Playwright MCP Server

This example demonstrates how to use the [Playwright MCP Server](https://github.com/microsoft/playwright-mcp) to control Playwright browsers through the Ailoy agent. You can instruct the agent to perform browser automation tasks, such as:

* Navigating pages and retrieving content
* Automating actions like clicks or keystrokes
* And more

> ⚠️ This example uses an OpenAI model, so make sure the `OPENAI_API_KEY` environment variable is set.

> ⚠️ The Playwright MCP server can quickly consume a large number of tokens due to the huge size of HTML page content, potentially leading to high costs if not used carefully.


### Prerequisites

* [Node.js](https://nodejs.org/ko/download)
* [Playwright](https://playwright.dev/)
  (Make sure to install the required browsers as explained in the [Playwright browser installation guide](https://playwright.dev/docs/browsers#install-browsers)):

  ```bash
  npx playwright install
  ```

### How to Run

```bash
# Set the `OPENAI_API_KEY` environment variable to use OpenAI models
export OPENAI_API_KEY=xxx
uv run playwright.py
```
