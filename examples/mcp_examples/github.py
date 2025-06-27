import os

from mcp import StdioServerParameters
import ailoy as ai


def main():
    rt = ai.Runtime()

    github_pat = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", None)
    if github_pat is None:
        github_pat = input("Enter GITHUB_PERSONAL_ACCESS_TOKEN: ")

    with ai.Agent(rt, ai.LocalModel("Qwen/Qwen3-8B")) as agent:
        agent.add_tools_from_mcp_server(
            "github",
            StdioServerParameters(
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server",
                ],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_pat,
                    # You can whitelist toolsets you want to use only.
                    # https://github.com/github/github-mcp-server?tab=readme-ov-file#available-toolsets
                    "GITHUB_TOOLSETS": "repos",
                },
            ),
            tools_to_add=[
                # You can whitelist tools you want to register to agent.
                # https://github.com/github/github-mcp-server?tab=readme-ov-file#tools
                "search_repositories",
                "get_file_contents",
            ],
        )

        print('Ailoy Github MCP Agent (Please type "exit" to stop conversation)')

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
