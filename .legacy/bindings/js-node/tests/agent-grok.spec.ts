import sharp from "sharp";

import * as ai from "../src";

const testImageUrl =
  "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";
const testAudioUrl =
  "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav";

if (process.env.XAI_API_KEY !== undefined) {
  describe("Grok Agent", () => {
    let rt: ai.Runtime;
    let agent: ai.Agent;

    before(async () => {
      rt = await ai.startRuntime();
      agent = await ai.defineAgent(
        rt,
        ai.APIModel({
          id: "grok-4",
          apiKey: process.env.XAI_API_KEY!,
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
        ai.ImageContent.fromUrl(testImageUrl),
      ])) {
        agent.print(resp);
      }
    });

    it("Image input from Sharp", async () => {
      const resp = await fetch(testImageUrl);
      const arrayBuffer = await resp.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const image = sharp(buffer);
      for await (const resp of agent.query(["What is in this image?", image])) {
        agent.print(resp);
      }
    });

    after(async () => {
      await agent.delete();
      await rt.stop();
    });
  });
}
