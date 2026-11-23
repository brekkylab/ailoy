import asyncio
import json

from ailoy import Agent, AgentBuilder
from ailoy.types import AgentSpec

# An agent as data: store it as JSON, load it, build it
spec: AgentSpec = {
    "model": "anthropic/claude-sonnet-5-5",
    "instruction": "Be brief.",
    "model_options": {"thinking_effort": "low"},
}
saved = json.dumps(spec)


async def main() -> None:
    agent = await Agent.from_spec(json.loads(saved))
    await agent.close()

    # In Python, the browser and context settings are the builder's keyword arguments
    browsing = await (
        AgentBuilder("anthropic/claude-sonnet-5-5")
        .browser(isolated=True, headless=True)
        .context_manager(max_input_tokens=100_000, preserve_recent_turns=2)
        .build()
    )
    await browsing.close()


asyncio.run(main())
