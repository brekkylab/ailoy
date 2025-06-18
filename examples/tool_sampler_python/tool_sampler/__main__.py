import os

from ailoy import Runtime, Agent
from ailoy.agent import BearerAuthenticator
import questionary


def main():
    # Start the Ailoy runtime
    rt = Runtime()

    # Select and initialize an agent (model)
    model_name: str = questionary.select(
        "Select the model",
        choices=[
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-0.6B",
            "o4-mini",
            "o3",
            "o3-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ],
    ).ask()
    if model_name.startswith("gpt") or model_name in ["o4-mini", "o3", "o3-mini"]:
        if "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = questionary.password("Enter OpenAI API key:").ask()
        if not api_key:
            raise RuntimeError("OpenAI API key not specified")
        agent = Agent(rt, model_name, api_key=api_key)
    else:
        agent = Agent(rt, model_name)

    # Select and configure tool to use
    preset_name: str = questionary.select(
        "Select the tool",
        choices=[
            questionary.Choice(
                title="Frankfurter",
                value="frankfurter",
                description="Real-time currency API (https://frankfurter.dev/)",
            ),
            questionary.Choice(
                title="The Movie Database (TMDB)",
                value="TMDB",
                description="Information about the movie (https://developer.themoviedb.org/reference/intro/getting-started)",
            ),
            questionary.Choice(
                title="New York Times",
                value="nytimes",
                description="New York Times articles (https://developer.nytimes.com/apis)",
            ),
        ],
    ).ask()

    if preset_name == "TMDB":
        api_key = os.getenv("TMDB_API_KEY")
        if api_key is None:
            api_key = questionary.password("Enter TMDB API key:").ask()
        agent.add_tools_from_preset("tmdb", authenticator=BearerAuthenticator(api_key))
    elif preset_name == "nytimes":
        api_key = os.getenv("NYTIMES_API_KEY")
        if api_key is None:
            api_key = questionary.password("Enter NYTimes API key: ").ask()

        def nytimes_authenticator(request):
            from urllib.parse import parse_qsl, urlencode, urlparse

            parts = urlparse(request.get("url", ""))
            qs = {**dict(parse_qsl(parts.query)), "api-key": api_key}
            parts = parts._replace(query=urlencode(qs))
            return {**request, "url": parts.geturl()}

        agent.add_tools_from_preset(
            "nytimes",
            authenticator=nytimes_authenticator,
        )
    else:
        agent.add_tools_from_preset(preset_name)

    # Ask whether reasoning mode should be enabled
    reasoning: str = questionary.text(
        "Do you want to enable reasoning? (Please type 'y' to enable):"
    ).ask()
    reasoning: bool = reasoning.lower() == "y"

    # Start conversation loop
    while True:
        query = questionary.text(
            'User (Please type "exit" to stop conversation):'
        ).ask()
        if query == "exit":
            break
        if query == "":
            continue
        for resp in agent.query(query, reasoning=reasoning):
            agent.print(resp)

    agent.delete()
    rt.stop()


if __name__ == "__main__":
    main()
