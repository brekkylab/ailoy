import asyncio

from ailoy._core import LocalLanguageModel, Message, Part, Role


async def main():
    user_message = Message(role=Role.User)
    user_message.append_content(Part.Text("What's the temperature of Seoul now?"))

    tool_call_message = Message(role=Role.Assistant)
    tool_call_message.append_tool_call(
        Part.Function(
            id="tool_call_0", name="temperature", arguments='{"location":"Seoul"}'
        )
    )

    tool_result_message = Message(role=Role.Tool)
    tool_result_message.append_content(
        Part.Text('{"temperature":"36","unit":"Celsius"}')
    )

    model = None
    async for v in LocalLanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v.comment, v.current, v.total)
        if v.result:
            model = v.result

    async for resp in model.run([user_message, tool_call_message, tool_result_message]):
        msg = resp.delta
        if msg.reasoning:
            print(f"\033[33m{msg.reasoning}\033[0m", end="", flush=True)
        else:
            for c in msg.content:
                print(c.text, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
