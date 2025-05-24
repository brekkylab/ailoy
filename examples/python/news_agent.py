import asyncio
import os

from ailoy import Runtime, Agent


async def main():
    rt = Runtime()

    nytimes_api_key = os.environ.get("NYTIMES_API_KEY", None)
    if nytimes_api_key is None:
        nytimes_api_key = input("Enter New York Times API Key: ")

    with Agent(rt, model_name="Qwen/Qwen3-8B") as agent:

        def nytimes_authenticator(request):
            from urllib.parse import parse_qsl, urlencode, urlparse

            parts = urlparse(request.get("url", ""))
            qs = {**dict(parse_qsl(parts.query)), "api-key": nytimes_api_key}
            parts = parts._replace(query=urlencode(qs))
            return {**request, "url": parts.geturl()}

        agent.add_tools_from_preset(
            "nytimes",
            authenticator=nytimes_authenticator,
        )

        print('Simple news assistant (Please type "exit" to stop conversation)')

        while True:
            query = input("\nUser: ")

            if query == "exit":
                break

            if query == "":
                continue

            for resp in agent.query(query):
                agent.print(resp)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
