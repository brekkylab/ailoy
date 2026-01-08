import json

import ailoy as ai
import gradio as gr


def frankfurter(base: str, symbols: str):
    """
    Get the latest currency exchange rates of target currencies based on the 'base' currency

    Args:
        base: The ISO 4217 currency code to be the divider of the currency rate to be got.
        symbols: The target ISO 4217 currency codes separated by comma.
    """
    import urllib.parse
    import urllib.request
    import urllib.error

    if not base:
        raise ValueError("Missing 'base'")
    if not symbols:
        raise ValueError("Missing 'symbols'")

    query = urllib.parse.urlencode({"from": base, "to": symbols})
    url = f"https://api.frankfurter.app/latest?{query}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Frankfurter API returned HTTP {resp.status}")
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Frankfurter API: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse Frankfurter API response as JSON.") from e

    return payload


with gr.Blocks() as demo:
    model = ai.LangModel.new_local_sync("Qwen/Qwen3-8B")
    agent = ai.Agent(model)
    tool = ai.Tool.new_py_function(frankfurter)
    agent.add_tool(tool)

    gr.Markdown("# Chat with Ailoy Agent")
    chatbot = gr.Chatbot(label="Agent", height="80vh")
    input = gr.Textbox(lines=1, label="Chat Message")

    def user_prompt(prompt, history):
        return "", history + [gr.ChatMessage(role="user", content=prompt)]

    def convert_messages_gr2ai(messages: list[gr.ChatMessage]) -> list[ai.Message]:
        def convert_message_delta(message: gr.ChatMessage) -> ai.MessageDelta:
            delta = ai.MessageDelta(role=message["role"])
            for option in message["options"]:
                if option["label"] == "type":
                    if option["value"] == "tool_call":
                        delta.tool_calls = [
                            ai.PartDelta.Function(
                                id=None,
                                function=ai.PartDeltaFunction.Verbatim(
                                    message["content"]
                                ),
                            )
                        ]
                        return delta

                    elif option["value"] == "tool_result":
                        delta.role = "tool"
                        delta.contents = [
                            ai.PartDelta.Value(json.loads(message["content"]))
                        ]
                        return delta

                    elif option["value"] == "thinking":
                        delta.thinking = message["content"][0]["text"]
                        return delta

            if len(message["content"]) > 0:
                delta.contents = [ai.PartDelta.Text(message["content"][0]["text"])]
            return delta

        deltas = list(map(convert_message_delta, messages))
        ai_messages: list[ai.Message] = []
        current = deltas[0]
        for next_delta in deltas[1:]:
            try:
                # try to merge the next delta into current
                current += next_delta
            except Exception:
                # if merge fails, convert current to Message and start fresh
                ai_messages.append(current.finish())
                current = next_delta

        # add the last accumulated delta
        ai_messages.append(current.finish())

        return ai_messages

    def convert_message_ai2gr(
        message: ai.Message | ai.MessageDelta,
    ) -> list[gr.ChatMessage]:
        msgs: list[gr.ChatMessage] = []

        # thinking
        if message.thinking is not None:
            msg = gr.ChatMessage(role="assistant", content=message.thinking)
            msg.metadata = {"title": "🧠 Thinking"}
            msg.options.append({"label": "type", "value": "thinking"})
            msgs.append(msg)

        # tool calls
        for tool_call in message.tool_calls:
            msg = gr.ChatMessage(role="assistant", content="")
            if isinstance(tool_call, ai.Part.Function):
                func = tool_call.function
                msg.content = json.dumps(
                    {"name": func.name, "arguments": func.arguments}
                )
                msg.metadata = {"title": f"🛠️ Tool Call: **{func.name}**"}
                msg.options.append({"label": "type", "value": "tool_call"})
            elif isinstance(tool_call, ai.PartDelta.Function):
                func = tool_call.function
                if isinstance(func, ai.PartDeltaFunction.Verbatim):
                    msg.content = func.text
            msgs.append(msg)

        # text content
        for content in message.contents:
            msg = gr.ChatMessage(role="assistant", content="")
            if isinstance(content, ai.PartDelta.Text) or isinstance(
                content, ai.Part.Text
            ):
                msg.content = content.text
            elif isinstance(content, ai.PartDelta.Value) or isinstance(
                content, ai.Part.Value
            ):
                msg.content = json.dumps(content.value)

            # if role is "tool", this content is considered as tool result
            if message.role == "tool":
                msg.metadata = {"title": "📄 Tool Result"}
                msg.options.append({"label": "type", "value": "tool_result"})

            msgs.append(msg)

        return msgs

    def agent_answer(messages: list[gr.ChatMessage]):
        converted_messages = convert_messages_gr2ai(messages)
        delta_acc = ai.MessageDelta()
        for resp in agent.run_delta_sync(
            converted_messages,
            config=ai.AgentConfig(
                inference=ai.LangModelInferConfig(think_effort="enable")
            ),
        ):
            delta_acc += resp.delta
            if resp.finish_reason:
                message = delta_acc.finish()
                messages.extend(convert_message_ai2gr(message))
                delta_acc = ai.MessageDelta()
                yield messages
            else:
                yield messages + convert_message_ai2gr(delta_acc)

    input.submit(user_prompt, [input, chatbot], [input, chatbot], queue=False).then(
        agent_answer, chatbot, chatbot
    )

    demo.launch()
