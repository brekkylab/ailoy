"use client";

import { useState, useEffect, type ReactNode } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  useExternalMessageConverter,
  CompositeAttachmentAdapter,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  type AppendMessage,
} from "@assistant-ui/react";
import * as ai from "ailoy-web";

export function AiloyRuntimeProvider({
  children,
}: Readonly<{ children: ReactNode }>) {
  const [agent, setAgent] = useState<ai.Agent | undefined>(undefined);
  const [agentLoading, setAgentLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<
    (ai.UserMessage | ai.AgentResponse)[]
  >([]);
  const [isAnswering, setIsAnswering] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      const { supported, reason } = await ai.isWebGPUSupported();
      if (!supported) {
        alert(`WebGPU is not supported: ${reason!}`);
        return;
      }

      setAgentLoading(true);
      const runtime = await ai.startRuntime();
      const agent = await ai.defineAgent(
        runtime,
        ai.LocalModel({ id: "Qwen/Qwen3-0.6B" })
        // ai.APIModel({
        //   id: "gpt-5-mini",
        //   apiKey: "<OPENAI_API_KEY>",
        // })
      );
      setAgent(agent);
      setAgentLoading(false);
    })();
  }, []);

  const onNew = async (message: AppendMessage) => {
    if (agent === undefined) throw new Error("Agent is not initialized yet");

    let userContent: ai.UserMessage["content"] = [];

    // Add attachments
    if (message.attachments !== undefined) {
      for (const attach of message.attachments) {
        if (attach.type === "image") {
          const imageContent = await ai.ImageContent.fromFile(attach.file!);
          userContent.push(imageContent);
        }
        // other types are skipped
      }
    }

    // Add text prompt
    if (message.content[0]?.type !== "text")
      throw new Error("Only text messages are supported");
    const textContent: ai.TextContent = {
      type: "text",
      text: message.content[0].text,
    };
    userContent.push(textContent);

    console.log(userContent);

    // Set messages
    setMessages((prev) => [...prev, { role: "user", content: userContent }]);
    setIsAnswering(true);

    for await (const resp of agent.query(userContent)) {
      if (resp.type === "output_text" || resp.type === "reasoning") {
        if (resp.isTypeSwitched) {
          setMessages((prev) => [...prev, resp]);
        } else {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            last.content += resp.content;
            return [...prev.slice(0, -1), last];
          });
        }
      } else {
        setMessages((prev) => [...prev, resp]);
      }
    }
    setIsAnswering(false);
  };

  const convertedMessages = useExternalMessageConverter({
    messages,
    callback: (message: ai.UserMessage | ai.AgentResponse) => {
      if (message.role === "user") {
        if (typeof message.content === "string") {
          return {
            role: message.role,
            content: [{ type: "text", text: message.content }],
          };
        } else {
          return {
            role: message.role,
            content: message.content.map((c) => {
              if (c.type === "text") return c;
              else if (c.type === "image_url")
                return { type: "image", image: c.image_url.url };
              else if (c.type === "input_audio")
                return { type: "audio", audio: c.input_audio };
              else throw Error("Unknown content type");
            }),
          };
        }
      } else if (message.type === "output_text") {
        return {
          role: "assistant",
          content: [{ type: "text", text: message.content }],
        };
      } else if (message.type === "reasoning") {
        return {
          role: "assistant",
          content: [{ type: "reasoning", text: message.content }],
        };
      } else if (message.type === "tool_call") {
        return {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: message.content.id!,
              toolName: message.content.function.name,
              args: message.content.function.arguments,
            },
          ],
        };
      } else if (message.type === "tool_call_result") {
        return {
          role: "tool",
          toolCallId: message.content.tool_call_id!,
          result: message.content.content[0].text,
        };
      } else {
        throw new Error(`Unknown message type: ${message.type}`);
      }
    },
    isRunning: isAnswering,
    joinStrategy: "concat-content",
  });

  const runtime = useExternalStoreRuntime({
    isLoading: agentLoading,
    isDisabled: agent === undefined,
    isRunning: isAnswering,
    messages: convertedMessages,
    onNew,
    adapters: {
      attachments: new CompositeAttachmentAdapter([
        new SimpleImageAttachmentAdapter(),
        new SimpleTextAttachmentAdapter(),
      ]),
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}
