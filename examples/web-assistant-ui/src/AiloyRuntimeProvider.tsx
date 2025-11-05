"use client";

import { useState, useEffect, useMemo, type ReactNode } from "react";
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

const imageDataToBase64 = (arr: Uint8Array): string => {
  let binaryString = "";
  arr.forEach((byte) => {
    binaryString += String.fromCharCode(byte);
  });
  const base64String = btoa(binaryString);
  return base64String;
};

function convertMessage(
  message: ai.Message
): useExternalMessageConverter.Message {
  if (message.role === "user") {
    return {
      role: message.role,
      content: message.contents.map((part) => {
        if (part.type === "text") return part;
        else if (part.type === "image") {
          if (part.image.type === "binary") {
            return {
              type: "image",
              image: `data:image/png;base64,${imageDataToBase64(
                part.image.data
              )}`,
            };
          } else {
            return { type: "image", image: part.image.url };
          }
        } else if (part.type === "value")
          return { type: "text", text: part.value!.toString() };
        else throw Error("Unknown content type");
      }),
    };
  } else if (message.role === "assistant") {
    let contents = [];
    if (message.thinking) {
      contents.push({
        type: "reasoning",
        text: message.thinking,
      });
    }
    if (message.tool_calls) {
      for (const toolCall of message.tool_calls) {
        if (toolCall.type !== "function")
          throw new Error("tool call content should be a type of function");
        contents.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          args: toolCall.function.arguments,
        });
      }
    }
    for (const content of message.contents) {
      if (content.type === "text") {
        contents.push(content);
      }
    }
    return {
      role: message.role,
      content: contents,
    } as useExternalMessageConverter.Message;
  } else if (message.role === "tool") {
    let toolResult: string;
    if (message.contents[0].type === "text") {
      toolResult = message.contents[0].text;
    } else if (message.contents[0].type === "value") {
      toolResult = JSON.stringify(message.contents[0].value);
    } else {
      throw new Error("Tool result should be either text or value.");
    }
    return {
      role: "tool",
      toolCallId: message.id,
      result: toolResult,
    } as useExternalMessageConverter.Message;
  } else {
    throw new Error(`Unknown message type: ${message}`);
  }
}

function convertMessageDelta(
  delta: ai.MessageDelta
): useExternalMessageConverter.Message {
  if (delta.role === "assistant") {
    let content = null;
    if (delta.thinking !== undefined) {
      content = {
        type: "reasoning",
        text: delta.thinking,
      };
    } else if (delta.tool_calls.length > 0) {
      for (const toolCall of delta.tool_calls) {
        if (toolCall.type !== "function")
          throw new Error("tool call content should be a type of function");
        if (toolCall.function.type === "verbatim") {
          content = { type: "text", text: toolCall.function.text };
        } else if (toolCall.function.type === "with_string_args") {
          content = {
            type: "text",
            text: `{"name": "${toolCall.function.name}", "arguments": ${toolCall.function.arguments}}`,
          };
        } else {
          content = {
            type: "text",
            text: `{"name": "${toolCall.function.name}", "arguments": ${toolCall.function.arguments}}`,
          };
        }
      }
    } else if (delta.contents.length > 0 && delta.contents[0].type === "text") {
      content = {
        type: "text",
        text: delta.contents[0].text,
      };
    }
    return {
      role: "assistant",
      content: content !== null ? [content] : [],
    } as useExternalMessageConverter.Message;
  } else if (delta.role === "tool") {
    let toolResult: string;
    if (delta.contents[0].type === "text") {
      toolResult = delta.contents[0].text;
    } else if (delta.contents[0].type === "value") {
      toolResult = JSON.stringify(delta.contents[0].value);
    } else {
      throw new Error("Tool result should be either text or value.");
    }
    return {
      role: "tool",
      toolCallId: delta.id,
      result: toolResult,
    } as useExternalMessageConverter.Message;
  } else {
    // Consider this case as an empty assistant message
    return {
      role: "assistant",
      content: [],
    };
  }
}

export function AiloyRuntimeProvider({
  children,
}: Readonly<{ children: ReactNode }>) {
  const [agent, setAgent] = useState<ai.Agent | undefined>(undefined);
  const [agentLoading, setAgentLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<ai.Message[]>([]);
  const [ongoingMessage, setOngoingMessage] = useState<ai.MessageDelta | null>(
    null
  );
  const [isAnswering, setIsAnswering] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      const { supported, reason } = await ai.isWebGPUSupported();
      if (!supported) {
        alert(`WebGPU is not supported: ${reason!}`);
        return;
      }

      setAgentLoading(true);
      const agent = new ai.Agent(
        await ai.LangModel.newLocal("Qwen/Qwen3-0.6B")
        // await ai.LangModel.newStreamAPI(
        //   "OpenAI",
        //   "gpt-4o",
        //   "<YOUR-OPENAI-API-KEY>"
        // )
      );

      setAgent(agent);
      setAgentLoading(false);
    })();
  }, []);

  const onNew = async (message: AppendMessage) => {
    if (agent === undefined) throw new Error("Agent is not initialized yet");

    let userContents: ai.Part[] = [];

    // Add attachments
    if (message.attachments !== undefined) {
      for (const attach of message.attachments) {
        if (attach.type === "image") {
          const ab = await attach.file!.arrayBuffer();
          const arr = new Uint8Array(ab);
          const imagePart = ai.imageFromBytes(arr);
          userContents.push(imagePart);
        }
        // other types are skipped
      }
    }

    // Add text prompt
    if (message.content[0]?.type !== "text")
      throw new Error("Only text messages are supported");
    userContents.push({ type: "text", text: message.content[0].text });

    // Set messages
    const newMessage: ai.Message = {
      role: "user",
      contents: userContents,
    };
    setMessages((prev) => [...prev, newMessage]);
    setIsAnswering(true);

    let accumulated: ai.MessageDelta | null = null;
    for await (const { delta, finish_reason } of agent.runDelta([
      ...messages,
      newMessage,
    ])) {
      accumulated =
        accumulated === null
          ? delta
          : ai.accumulateMessageDelta(accumulated, delta);
      setOngoingMessage({ ...accumulated });

      if (finish_reason !== undefined) {
        let newMessage = ai.finishMessageDelta(accumulated);
        setMessages((prevMessages) => [...prevMessages, newMessage]);
        setOngoingMessage(null);
        accumulated = null;
      }
    }
    setIsAnswering(false);
  };

  const convertedMessages: useExternalMessageConverter.Message[] =
    useMemo(() => {
      let converted = messages.map(convertMessage);
      if (ongoingMessage !== null) {
        let convertedDelta = convertMessageDelta(ongoingMessage);
        converted = [...converted, convertedDelta];
      }
      return converted;
    }, [messages, ongoingMessage]);

  const runtime = useExternalStoreRuntime({
    isLoading: agentLoading,
    isDisabled: agent === undefined,
    isRunning: isAnswering,
    messages: useExternalMessageConverter({
      messages: convertedMessages,
      callback: (msg) => msg,
      isRunning: isAnswering,
    }),
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
