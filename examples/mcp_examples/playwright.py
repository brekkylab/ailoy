import asyncio
import os
import sys

import ailoy as ai
from aioconsole import ainput


async def main():
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_api_key is None:
        openai_api_key = input("Enter OPENAI_API_KEY: ")

    model = ai.LangModel.new_stream_api("OpenAI", "gpt-5.2", openai_api_key)
    agent = ai.Agent(model)

    mcp_client = await ai.MCPClient.from_stdio("npx", ["@playwright/mcp@latest"])
    agent.add_tools(
        [
            mcp_client.get_tool("browser_click"),
            mcp_client.get_tool("browser_navigate"),
        ]
    )

    print('Ailoy Playwright MCP Agent (Please type "exit" to stop conversation)')

    os.set_blocking(sys.stdout.fileno(), True)

    try:
        messages: list[ai.Message] = []
        while True:
            query = await ainput("\n\nUser: ")

            if query == "exit":
                break

            if query == "":
                continue

            messages.append(ai.Message(role="user", contents=query))

            acc = ai.MessageDelta()
            async for resp in agent.run_delta(messages):
                acc += resp.delta
                if resp.finish_reason:
                    message = acc.finish()
                    acc = ai.MessageDelta()
                    messages.append(message)

                for content in resp.delta.contents:
                    if isinstance(content, ai.PartDelta.Text):
                        print(content.text, end="", flush=True)
                    elif isinstance(content, ai.PartDelta.Value):
                        print(content.value[:100])
                    else:
                        raise ValueError(
                            f"Content has invalid part_type: {content.part_type}"
                        )
                for tool_call in resp.delta.tool_calls:
                    if isinstance(tool_call, ai.PartDelta.Function):
                        if isinstance(
                            tool_call.function, ai.PartDeltaFunction.Verbatim
                        ):
                            print(tool_call.function.text, end="", flush=True)
                        elif isinstance(
                            tool_call.function, ai.PartDeltaFunction.WithStringArgs
                        ):
                            print(tool_call.function.arguments, end="", flush=True)
                        elif isinstance(
                            tool_call.function, ai.PartDeltaFunction.WithParsedArgs
                        ):
                            print(tool_call.function.arguments, end="", flush=True)
                    else:
                        raise ValueError(
                            f"Tool call has invalid part_type: {tool_call.part_type}"
                        )

    except asyncio.exceptions.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
