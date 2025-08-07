import {
  expect,
  expectTypeOf,
  describe,
  beforeEach,
  afterAll,
  it,
} from "vitest";
import Vips from "wasm-vips";

import { defineAgent } from "../src/agent";
import { Runtime } from "../src/runtime";
import { APIModel } from "../src/models";

describe.skipIf(process.env.CLAUDE_API_KEY === "undefined")(
  "Claude Agent",
  async () => {
    const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY!;

    const runtime = new Runtime();
    await runtime.start();

    const agent = await defineAgent(
      runtime,
      APIModel({ id: "claude-sonnet-4-20250514", apiKey: CLAUDE_API_KEY })
    );

    beforeEach(() => {
      agent.clearMessages();
      agent.clearTools();
    });

    it.sequential("Hello World", async () => {
      const iter = agent.query("Hello World!");
      const resp = await iter.next();
      /**
       * Example response
        {
          "content": "Hi there! How can I assist you today?",
          "isTypeSwitched": true,
          "role": "assistant",
          "type": "output_text",
        }
      */
      expect(resp.value).to.have.property("type", "output_text");
      expect(resp.value).to.have.property("role", "assistant");
      expect(resp.value).to.have.property("content");
    });

    it.sequential("Tool Calling", async () => {
      await agent.addToolsFromPreset("frankfurter");
      expect(agent.getTools()).to.be.lengthOf(1);

      const tool = agent.getTools()[0];
      expect(tool.desc.name).toBe("frankfurter");

      const iter = agent.query(
        "How much is $100 in KRW? Answer in a number without commas"
      );

      // Claude returns a tool_call message after a single output_text message,
      // so we discard the first message and continue.
      await iter.next();

      const toolCallResp = await iter.next();
      /**
         * Example toolCallResp
          {
            "content": {
              "function": {
                "arguments": {
                  "base": "USD",
                  "symbols": "KRW",
                },
                "name": "frankfurter",
              },
              "id": "call_GHiqPOWq2KGMlntor4zkJtb6",
              "type": "function",
            },
            "isTypeSwitched": true,
            "role": "assistant",
            "type": "tool_call",
          }
        */
      expect(toolCallResp.value).to.have.property("content");
      expect(toolCallResp.value.content).to.have.property("type", "function");
      expect(toolCallResp.value.content).to.have.property("function");
      expect(toolCallResp.value.content.function).to.have.property(
        "name",
        "frankfurter"
      );
      expect(toolCallResp.value.content.function.arguments).to.have.property(
        "base",
        "USD"
      );
      expect(toolCallResp.value.content.function.arguments).to.have.property(
        "symbols",
        "KRW"
      );

      const toolCallResultResp = await iter.next();
      /**
         * Example toolCallResultResp
          {
            "content": {
              "content": [
                {
                  "text": "{"KRW":1404.62}",
                  "type": "text",
                },
              ],
              "role": "tool",
              "tool_call_id": "call_GHiqPOWq2KGMlntor4zkJtb6",
            },
            "isTypeSwitched": true,
            "role": "tool",
            "type": "tool_call_result",
          }
        */
      expect(toolCallResultResp.value).to.have.property("content");
      expect(toolCallResultResp.value).to.have.property("role", "tool");
      expect(toolCallResultResp.value.content).to.have.property("content");
      expectTypeOf(toolCallResultResp.value.content.content[0]).toMatchTypeOf<{
        type: "text";
        text: string;
      }>();
      const toolResult = JSON.parse(
        toolCallResultResp.value.content.content[0].text
      );
      expectTypeOf(toolResult).toMatchTypeOf<{ KRW: number }>();
      const krw: number = toolResult.KRW;

      let finalResponse = "";
      for await (const resp of iter) {
        finalResponse += resp.content;
      }
      // Answer should include 100 * KRW
      expect(finalResponse).toMatch(`${100 * krw}`);
    });

    it.sequential("Image Inputs", async () => {
      // Test image input
      const vips = await Vips();
      const imageResp = await fetch(
        "https://newsroom.haas.berkeley.edu/wp-content/uploads/2023/02/jensen-huang-headshot2_thmb-300x246.jpg"
      );
      const arrayBuffer = await imageResp.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      const image = vips.Image.newFromBuffer(uint8Array);

      const iter = agent.query([image, "What is in this image?"]);
      const resp = await iter.next();
      /**
       * Example response
        {
          "content": "The image shows a person wearing glasses and a leather jacket, smiling at the camera.",
          "isTypeSwitched": true,
          "role": "assistant",
          "type": "output_text",
        }
      */
      expect(resp.value).to.have.property("type", "output_text");
      expect(resp.value).to.have.property("role", "assistant");
      expect(resp.value).to.have.property("content");
      expect(resp.value.content).toMatch(/glasses/);
      expect(resp.value.content).toMatch(/jacket/);
    });

    afterAll(async () => {
      await agent.delete();
      await runtime.stop();
    });
  }
);
