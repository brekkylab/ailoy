import json

import ailoy as ai
import gradio as gr


rt = ai.Runtime()

with gr.Blocks() as demo, ai.Agent(rt, "Qwen/Qwen3-8B") as agent:
    agent.add_tools_from_preset("frankfurter")

    gr.Markdown("# Chat with Ailoy Agent")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
    )
    input = gr.Textbox(lines=1, label="Chat Message")

    def user_prompt(prompt, history):
        return "", history + [gr.ChatMessage(role="user", content=prompt)]

    def agent_answer(messages):
        user_prompt = messages[-1]["content"]
        for resp in agent.query(user_prompt, enable_reasoning=True):
            if resp.type == "tool_call":
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=json.dumps(resp.content.function.arguments),
                        metadata={
                            "title": f"🛠️ Tool Call: {resp.content.function.name}({resp.content.id})"
                        },
                    )
                )
            elif resp.type == "tool_call_result":
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=resp.content.content,
                        metadata={
                            "title": f"📄 Tool Result: {resp.content.name}({resp.content.tool_call_id})"
                        },
                    )
                )
            elif resp.type == "reasoning":
                if resp.is_type_switched:
                    messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": "🧠 Thinking"},
                        )
                    )
                messages[-1].content += resp.content
            elif resp.type == "output_text":
                if resp.is_type_switched:
                    messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content="",
                        )
                    )
                messages[-1].content += resp.content

            yield messages

    input.submit(user_prompt, [input, chatbot], [input, chatbot], queue=False).then(
        agent_answer, chatbot, chatbot
    )

    demo.launch()
