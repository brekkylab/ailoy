import sharp from "sharp";

import * as ai from "../src";

const testImageUrl =
  "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";
const testAudioUrl =
  "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav";

if (process.env.GEMINI_API_KEY !== undefined) {
  describe("Gemini Agent", () => {
    let rt: ai.Runtime;
    let agent: ai.Agent;

    before(async () => {
      rt = await ai.startRuntime();
      agent = await ai.defineAgent(
        rt,
        ai.APIModel({
          id: "gemini-2.5-flash",
          apiKey: process.env.GEMINI_API_KEY!,
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

    it("Image recognition from base64", async () => {
      const resp = await fetch(testImageUrl);
      const arrayBuffer = await resp.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const image = sharp(buffer);
      for await (const resp of agent.query([
        "What is in this image?",
        await ai.ImageContent.fromSharp(image),
      ])) {
        agent.print(resp);
      }
    });

    it("Audio input from base64", async () => {
      const resp = await fetch(testAudioUrl);
      const arrayBuffer = await resp.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      for await (const resp of agent.query([
        "What's in these recording?",
        await ai.AudioContent.fromBytes(buffer, "wav"),
      ])) {
        agent.print(resp);
      }

      await agent.delete();
    });

    after(async () => {
      await agent.delete();
      await rt.stop();
    });
  });
}
