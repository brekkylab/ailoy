import asyncio
import os
import textwrap

from ailoy import AsyncRuntime, Agent
from ailoy.agent import BearerAuthenticator, AgentResponse


def print_agent_response(resp: AgentResponse):
    if resp.type == "output_text":
        print(resp.content, end="")
        if resp.end_of_turn:
            print("\n\n")
    elif resp.type == "reasoning":
        print(f"\033[93m{resp.content}\033[0m", end="")
        if resp.end_of_turn:
            print("\n\n")
    elif resp.type == "tool_call":
        print(
            textwrap.dedent(
                f"""
                Tool Call
                - ID: {resp.content.id}
                - name: {resp.content.function.name}
                - arguments: {resp.content.function.arguments}
                """
            )
        )
    elif resp.type == "tool_call_result":
        print(
            textwrap.dedent(
                f"""
                Tool Call Result
                - ID: {resp.content.tool_call_id}
                - name: {resp.content.name}
                - Result: {resp.content.content}
                """
            )
        )


async def main():
    rt = AsyncRuntime()

    tmdb_api_key = os.environ.get("TMDB_API_KEY", None)
    if tmdb_api_key is None:
        tmdb_api_key = input("Enter TMDB API Key: ")

    executor = Agent(rt, model_name="qwen3-8b")

    # Use openai model
    # openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    # if openai_api_key is None:
    #     openai_api_key = input("Enter OpenAI API Key: ")
    # executor = Agent(rt, model_name="gpt-4o", api_key=openai_api_key)

    await executor.initialize()

    executor.add_tools_from_preset(
        "tmdb", authenticator=BearerAuthenticator(tmdb_api_key)
    )

    print('Simple movie assistant (Please type "exit" to stop conversation)')

    while True:
        query = input("\nUser: ")

        if query == "exit":
            break

        if query == "":
            continue

        async for resp in executor.run(
            query,
            do_reasoning=True,
        ):
            print_agent_response(resp)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
