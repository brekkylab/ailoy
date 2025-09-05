import asyncio

from ailoy._core import (
    Agent,
    LocalLanguageModel,
)


async def main():
    # tool = ToolDesc(
    #     "temperature",
    #     "Get current temperature",
    #     {
    #         "type": "object",
    #         "properties": {
    #             "location": {"type": "string", "description": "The city name"},
    #             "unit": {
    #                 "type": "string",
    #                 "description": "Default: Celcius",
    #                 "enum": ["Celcius", "Fernheit"],
    #             },
    #         },
    #         "required": ["location"],
    #     },
    #     returns={"type": "number"},
    # )

    # user_message = Message(role=Role.User)
    # user_message.append_content(Part.Text("What's the temperature of Seoul now?"))

    # tool_call_message = Message(role=Role.Assistant)
    # tool_call_message.append_tool_call(
    #     Part.Function(
    #         id="tool_call_0", name="temperature", arguments='{"location":"Seoul"}'
    #     )
    # )

    # tool_result_message = Message(role=Role.Tool)
    # tool_result_message.tool_call_id = "tool_call_0"
    # tool_result_message.append_content(Part.Text('{"temperature": 38.5}'))

    model = None
    async for v in LocalLanguageModel.create("Qwen/Qwen3-0.6B"):
        print(v.comment, v.current, v.total)
        if v.result:
            model = v.result
            model.disable_reasoning()

    agent = Agent(model)
    async for resp in agent.run("What is your name?"):
        print(resp)
    # async for resp in agent.lm.run(
    #     [user_message, tool_call_message, tool_result_message], tools=[tool]
    # ):
    #     print(resp)


if __name__ == "__main__":
    asyncio.run(main())
