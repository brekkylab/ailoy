import sharp from "sharp";

import * as ai from "../src";

const testImageUrl =
  "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";
const testAudioUrl =
  "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav";

if (process.env.OPENAI_API_KEY !== undefined) {
  describe("OpenAI Agent", () => {
    let rt: ai.Runtime;
    let agent: ai.Agent;

    before(async () => {
      rt = await ai.startRuntime();
      agent = await ai.defineAgent(
        rt,
        ai.OpenAIModel({
          id: "gpt-4.1-mini",
          apiKey: process.env.OPENAI_API_KEY!,
        })
      );
    });

    beforeEach(() => {
      agent.clearMessages();
      agent.clearTools();
    });

    it("Tool Call: calculator tools", async () => {
      agent.addToolsFromPreset("calculator");

      const query =
        "Please calculate this formula: floor(ln(exp(e))+cos(2*pi))";
      console.log(`Query: ${query}`);

      for await (const resp of agent.query(query)) {
        agent.print(resp);
      }
    });

    it("Image input from URL", async () => {
      for await (const resp of agent.query([
        "What is in this image?",
        {
          type: "image_url",
          url: testImageUrl,
        },
      ])) {
        agent.print(resp);
      }
    });

    it("Image input from base64", async () => {
      const resp = await fetch(testImageUrl);
      const arrayBuffer = await resp.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const image = sharp(buffer);
      for await (const resp of agent.query([
        "What is in this image?",
        {
          type: "image_sharp",
          image,
        },
      ])) {
        agent.print(resp);
      }
    });

    it("Audio input from base64", async () => {
      const audioAgent = await ai.defineAgent(
        rt,
        ai.OpenAIModel({
          // OpenAI general models cannot handle audio inputs.
          id: "gpt-4o-audio-preview",
          apiKey: process.env.OPENAI_API_KEY!,
        })
      );

      const resp = await fetch(testAudioUrl);
      const arrayBuffer = await resp.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      for await (const resp of audioAgent.query([
        "What's in these recording?",
        {
          type: "audio_bytes",
          data: buffer,
          format: "wav",
        },
      ])) {
        audioAgent.print(resp);
      }

      await audioAgent.delete();
    });

    after(async () => {
      await agent.delete();
      await rt.stop();
    });
  });
}
