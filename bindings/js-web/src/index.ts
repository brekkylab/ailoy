import { GenerationConfig } from "./config";
import { Engine } from "./engine";

async function main() {
  const engine = new Engine("Qwen/Qwen3-0.6B");
  await engine.loadModel();

  const genConfig: GenerationConfig = {
    // temperature: 1.0,
  };

  const generator = (await engine.inferLM(
    [
      {
        role: "system",
        // content: [{type: "text", text: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}],
        content:
          "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
      },
      {
        role: "user",
        // content: [{type: "text", text: "How is the current weather in Seoul?"}],
        content: "How is the current weather in Seoul?",
      },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              arguments: { location: "Seoul, South Korea", unit: "celsius" },
              name: "get_current_temperature",
            },
            type: "function",
          },
        ],
      },
      {
        role: "tool",
        // content: [{ type: "text", text: "20.5" }],
        content: "20.5",
      },
    ],
    false,
    genConfig
  )) as AsyncGenerator<any, void, void>;
  console.log(generator);

  for await (const resp of generator) {
    console.log(resp.choices[0].delta.content, "");
    // Module.print(resp.choices[0].delta.content, "");
  }
}

main().then(() => {
  console.log("Good.");
});
