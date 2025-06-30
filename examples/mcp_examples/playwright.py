import os

import ailoy as ai
from mcp import StdioServerParameters


def main():
    rt = ai.Runtime()

    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        openai_api_key = input("Enter OPENAI_API_KEY: ")

    with ai.Agent(rt, ai.APIModel("gpt-4o", api_key=openai_api_key)) as agent:
        agent.add_tools_from_mcp_server(
            "playwright",
            # https://github.com/microsoft/playwright-mcp?tab=readme-ov-file#getting-started
            StdioServerParameters(command="npx", args=["@playwright/mcp@latest"]),
            tools_to_add=[
                # You can whitelist tools you want to register to agent.
                # https://github.com/microsoft/playwright-mcp?tab=readme-ov-file#tools
                "browser_click",
                "browser_navigate",
            ],
        )

        print('Ailoy Playwright MCP Agent (Please type "exit" to stop conversation)')

        while True:
            query = input("\n\nUser: ")

            if query == "exit":
                break

            if query == "":
                continue

            for resp in agent.query(query):
                agent.print(resp)


if __name__ == "__main__":
    main()
