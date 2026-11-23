import asyncio

from ailoy import AgentBuilder


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        # Whole messages: why each ended, and what it cost
        async for out in agent.run("Explain DNS in two sentences."):
            reason = out["finish_reason"]
            usage = out.get("usage")
            if usage:
                print("tokens in/out:", usage["input_tokens"], usage["output_tokens"])
            if reason["type"] == "refusal":
                print("refused:", reason.get("reason"))

        # Deltas: put the text back together until a finish_reason ends the message
        text = ""
        async for delta in agent.run_stream("Now explain it to a child."):
            for piece in delta["delta"]["contents"]:
                text += piece.get("text", "")
            done = delta["finish_reason"]
            if done:
                print(text, f"({done['type']})")
                text = ""


asyncio.run(main())
