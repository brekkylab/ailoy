import textwrap

from ailoy.agent import AgentResponse


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
