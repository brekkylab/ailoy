import sharp from "sharp";

import * as ai from "../src";

const testImageUrl =
  "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";

if (process.env.CLAUDE_API_KEY !== undefined) {
  describe("Claude Agent", () => {
    let rt: ai.Runtime;
    let agent: ai.Agent;

    before(async () => {
      rt = await ai.startRuntime();
      agent = await ai.defineAgent(
        rt,
        ai.ClaudeModel({
          id: "claude-sonnet-4-20250514",
          apiKey: process.env.CLAUDE_API_KEY!,
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
        {
          type: "image_sharp",
          image,
        },
      ])) {
        agent.print(resp);
      }
    });

    after(async () => {
      await agent.delete();
      await rt.stop();
    });
  });
}
