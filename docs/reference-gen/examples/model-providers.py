import asyncio

from ailoy import LangModel, get_lang_model_provider, set_lang_model_provider
from ailoy.types import LangModelOptions, Message, MessageOutput, ToolDesc

# Claude through its own API (reads ANTHROPIC_API_KEY) instead of Claude Code
set_lang_model_provider("anthropic", "api")
print(get_lang_model_provider("anthropic/claude-sonnet-5-5"))  # api

# Qwen on your own vLLM server (VLLM_BASE_URL)
set_lang_model_provider("Qwen", "vllm")


# A model of your own: here, one that repeats the last message back
class Echo:
    def infer(
        self, messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
    ) -> MessageOutput:
        last = messages[-1]["contents"]
        return {"message": {"role": "assistant", "contents": last}, "finish_reason": {"type": "stop"}}


set_lang_model_provider("echo", lambda org, name: Echo())


async def main() -> None:
    model = await LangModel.create("echo/any-name")
    out = await model.infer([{"role": "user", "contents": [{"type": "text", "text": "hi"}]}])
    print(out["message"]["contents"])


asyncio.run(main())
