import { search } from "jmespath";

import { Runtime, generateUUID } from "./runtime";

/** Types for OpenAI API-compatible data structures */

interface SystemMessage {
  role: "system";
  content: string;
}

interface UserMessage {
  role: "user";
  content: string;
}

interface AIOutputTextMessage {
  role: "assistant";
  content: string;
  reasoning?: boolean;
}

interface AIToolCallMessage {
  role: "assistant";
  content: null;
  tool_calls: Array<ToolCall>;
}

interface ToolCall {
  id: string;
  function: { name: string; arguments: any };
}

interface ToolCallResultMessage {
  role: "tool";
  name: string;
  tool_call_id: string;
  content: string;
}

type Message =
  | SystemMessage
  | UserMessage
  | AIOutputTextMessage
  | AIToolCallMessage
  | ToolCallResultMessage;

interface MessageDelta {
  finish_reason: "stop" | "tool_calls" | "length" | "error" | null;
  message: Message;
}

/** Types for LLM Model Definitions */

type TVMModel = "qwen3-8b" | "qwen3-4b" | "qwen3-1.7b" | "qwen3-0.6b";
interface TVMModelAttrs {
  name: TVMModel;
  quantization?: "q4f16_1";
  mode?: "interactive" | "local" | "server";
}

type OpenAIModel = "gpt-4o";
interface OpenAIModelAttrs {
  name: OpenAIModel;
  api_key: string;
}

interface ModelDescription {
  modelId: string;
  componentType: string;
  defaultSystemMessage?: string;
}

type AvailableModel = TVMModel | OpenAIModel;
const modelDescriptions: Record<AvailableModel, ModelDescription> = {
  "qwen3-8b": {
    modelId: "Qwen/Qwen3-8B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "qwen3-4b": {
    modelId: "Qwen/Qwen3-4B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "qwen3-1.7b": {
    modelId: "Qwen/Qwen3-1.7B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "qwen3-0.6b": {
    modelId: "Qwen/Qwen3-0.6B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "gpt-4o": {
    modelId: "gpt-4o",
    componentType: "openai",
  },
};

/** Types for Agent's responses */

interface AgentResponse_Base {
  type:
    | "output_text"
    | "tool_call"
    | "tool_call_result"
    | "reasoning"
    | "error";
  endOfTurn: boolean;
  role: "assistant" | "tool";
}
interface AgentResponse_OutputText extends AgentResponse_Base {
  type: "output_text" | "reasoning";
  role: "assistant";
  content: string;
}
interface AgentResponse_ToolCall extends AgentResponse_Base {
  type: "tool_call";
  role: "assistant";
  content: {
    id: string;
    function: { name: string; arguments: any };
  };
}
interface AgentResponse_ToolCallResult extends AgentResponse_Base {
  type: "tool_call_result";
  role: "tool";
  content: {
    name: string;
    tool_call_id: string;
    content: string;
  };
}
interface AgentResponse_Error extends AgentResponse_Base {
  type: "error";
  role: "assistant";
  error: string;
}
export type AgentResponse =
  | AgentResponse_OutputText
  | AgentResponse_ToolCall
  | AgentResponse_ToolCallResult
  | AgentResponse_Error;

/** Types and functions related to Tools */

interface Tool {
  desc: ToolDescription;
  call: (runtime: Runtime, input: any) => Promise<any>;
}

interface ToolDescription {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: {
      [key: string]: {
        type: "string" | "number" | "boolean" | "object" | "array" | "null";
        description?: string;
        [key: string]: any;
      };
    };
    required?: string[];
  };
  return?: {
    type: "string" | "number" | "boolean" | "object" | "array" | "null";
    description?: string;
    [key: string]: any;
  };
}

interface BaseToolDefinition {
  type: string;
  description: ToolDescription;
  behavior: Record<string, any>;
}

interface UniversalToolDefinition extends BaseToolDefinition {
  type: "universal";
  behavior: {
    outputPath?: string;
  };
}

interface RESTAPIToolDefinition extends BaseToolDefinition {
  type: "restapi";
  behavior: {
    baseURL: string;
    method: "GET" | "POST" | "PUT" | "DELETE";
    headers: { [key: string]: string };
    body?: string;
    outputPath?: string;
  };
}

type ToolDefinition = UniversalToolDefinition | RESTAPIToolDefinition;

export interface ToolAuthenticator {
  apply: (request: {
    url: string;
    headers: { [k: string]: string };
    [k: string]: any;
  }) => {
    url: string;
    headers: { [k: string]: string };
    [k: string]: any;
  };
}

export function bearerAutenticator(
  token: string,
  bearerFormat: string = "Bearer"
): ToolAuthenticator {
  return {
    apply: (request: {
      url: string;
      headers: { [k: string]: string };
      [k: string]: any;
    }) => {
      return {
        ...request,
        headers: {
          ...request.headers,
          Authorization: `${bearerFormat} ${token}`,
        },
      };
    },
  };
}

/** Assistant class */
export class Agent {
  private runtime: Runtime;
  private modelInfo: {
    componentType: string;
    componentName: string;
    attrs: Record<string, any>;
  };
  private tools: Tool[];
  private messages: Message[];
  private valid: boolean;

  constructor(
    runtime: Runtime,
    args: {
      model: TVMModelAttrs | OpenAIModelAttrs;
      tools?: Tool[];
      systemMessage?: string;
    }
  ) {
    this.runtime = runtime;

    // Model's short name (`modelName`) will be replaced with full model id defined in description.
    // The another attributes (`modelAttrs`) are used as-is.
    const { name: modelName, ...modelAttrs } = args.model;
    if (!(modelName in modelDescriptions))
      throw Error(`Model "${modelName}" is not available`);

    const modelDesc = modelDescriptions[modelName];
    this.modelInfo = {
      componentType: modelDesc.componentType,
      componentName: generateUUID(),
      attrs: {
        model: modelDesc.modelId,
        ...modelAttrs,
      },
    };

    this.messages = [];

    // Use system message provided from arguments first, otherwise use default system message for the model.
    let systemMessage =
      args.systemMessage !== undefined
        ? args.systemMessage
        : modelDesc.defaultSystemMessage;
    // Append system message if defined.
    if (systemMessage !== undefined) {
      this.messages.push({ role: "system", content: systemMessage });
    }

    this.tools = args.tools || [];

    this.valid = false;
  }

  addTool(tool: Tool): boolean {
    if (this.tools.find((t) => t.desc.name == tool.desc.name)) return false;
    this.tools.push(tool);
    return true;
  }

  addJSFunctionTool(desc: ToolDescription, f: (input: any) => any): boolean {
    return this.addTool({ desc, call: f });
  }

  addUniversalTool(tool: UniversalToolDefinition): boolean {
    const call = async (runtime: Runtime, inputs: any) => {
      // Validation
      const required = tool.description.parameters.required || [];
      const missing = required.filter((name) => !(name in inputs));
      if (missing.length != 0)
        throw Error("some parameters are required but not exist: " + missing);

      // Call
      let output = await runtime.call(tool.description.name, inputs);

      // Parse output path
      if (tool.behavior.outputPath)
        output = search(output, tool.behavior.outputPath);

      // Return
      return output;
    };
    return this.addTool({ desc: tool.description, call });
  }

  addRESTAPITool(
    tool: RESTAPIToolDefinition,
    auth?: ToolAuthenticator
  ): boolean {
    const call = async (runtime: Runtime, inputs: any) => {
      const { baseURL, method, headers, body, outputPath } = tool.behavior;
      const renderTemplate = (
        template: string,
        context: Record<string, string | number>
      ): {
        result: string;
        variables: string[];
      } => {
        const regex = /\$\{\s*([^}\s]+)\s*\}/g;
        const variables = new Set<string>();
        const result = template.replace(regex, (_, key: string) => {
          variables.add(key);
          return key in context ? (context[key] as string) : `{${key}}`;
        });
        return {
          result,
          variables: Array.from(variables),
        };
      };

      // Handle path parameters
      const { result: pathHandledURL, variables: pathParameters } =
        renderTemplate(baseURL, inputs);

      // Define URL
      const url = new URL(pathHandledURL);

      // Handle body
      const { result: bodyRendered, variables: bodyParameters } = body
        ? renderTemplate(body, inputs)
        : { result: undefined, variables: [] };

      // Default goes to query parameters
      let qp: [string, string][] = [];
      Object.entries(inputs).forEach(([key, _]) => {
        const findResult1 = pathParameters.find((v) => v === key);
        const findResult2 = bodyParameters.find((v) => v === key);
        if (!findResult1 && !findResult2) qp.push([key, inputs[key]]);
      });
      url.search = new URLSearchParams(qp).toString();

      // Create request
      const requestNoAuth = bodyRendered
        ? {
            url: url.toString(),
            method,
            headers,
            body: bodyRendered,
          }
        : {
            url: url.toString(),
            method,
            headers,
          };

      // Apply authentication
      const request = auth ? auth.apply(requestNoAuth) : requestNoAuth;

      // Call
      let output: any;
      const resp = await runtime.call("http_request", request);
      // @jhlee: How to parse it?
      output = JSON.parse(resp.body);

      // Parse output path
      if (outputPath) output = search(output, outputPath);

      // Return
      return output;
    };
    return this.addTool({ desc: tool.description, call });
  }

  addToolsFromPreset(
    presetName: string,
    args?: { authenticator?: ToolAuthenticator }
  ): boolean {
    const presetJson = require(`./presets/tools/${presetName}.json`);
    if (presetJson === undefined) {
      throw Error(`Preset "${presetName}" does not exist`);
    }

    for (const tool of Object.values(
      presetJson as Record<string, ToolDefinition>
    )) {
      if (tool.type == "restapi") {
        const result = this.addRESTAPITool(tool, args?.authenticator);
        if (!result) return false;
      } else if (tool.type == "universal") {
        const result = this.addUniversalTool(tool);
        if (!result) return false;
      }
    }
    return true;
  }

  getMessages(): Message[] {
    return this.messages;
  }

  setMessages(messages: Message[]) {
    this.messages = messages;
  }

  appendMessage(msg: Message) {
    this.messages.push(msg);
  }

  async initialize(): Promise<void> {
    if (this.valid) return;
    const result = await this.runtime.define(
      this.modelInfo.componentType,
      this.modelInfo.componentName,
      this.modelInfo.attrs
    );
    if (!result) throw Error(`model initializeation failed`);
    this.valid = true;
  }

  async *run(
    message: string,
    options?: {
      reasoning?: boolean;
      ignore_reasoning?: boolean;
    }
  ): AsyncGenerator<AgentResponse> {
    this.messages.push({ role: "user", content: message });

    while (true) {
      for await (const resp of this.runtime.callIterMethod(
        this.modelInfo.componentName,
        "infer",
        {
          messages: this.messages,
          tools: this.tools.map((v) => {
            return { type: "function", function: v.desc };
          }),
          reasoning: options?.reasoning,
          ignore_reasoning: options?.ignore_reasoning,
        }
      )) {
        const delta: MessageDelta = resp;

        // This means AI is still streaming tokens
        if (delta.finish_reason === null) {
          let message = delta.message as AIOutputTextMessage;
          yield {
            type: message.reasoning ? "reasoning" : "output_text",
            endOfTurn: false,
            role: "assistant",
            content: message.content,
          };
          continue;
        }

        // This means AI requested tool calls
        if (delta.finish_reason === "tool_calls") {
          const toolCallMessage = delta.message as AIToolCallMessage;
          // Add tool call back to messages
          this.messages.push(toolCallMessage);

          // Yield for each tool call
          for (const toolCall of toolCallMessage.tool_calls) {
            yield {
              type: "tool_call",
              endOfTurn: true,
              role: "assistant",
              content: toolCall,
            };
          }

          // Call tools in parallel
          let toolCallPromises: Array<Promise<ToolCallResultMessage>> = [];
          for (const toolCall of toolCallMessage.tool_calls) {
            toolCallPromises.push(
              new Promise(async (resolve, reject) => {
                const tool_ = this.tools.find(
                  (v) => v.desc.name == toolCall.function.name
                );
                if (!tool_) {
                  reject("Internal exception");
                  return;
                }
                const resp = await tool_.call(
                  this.runtime,
                  toolCall.function.arguments
                );
                const message: ToolCallResultMessage = {
                  role: "tool",
                  name: toolCall.function.name,
                  tool_call_id: toolCall.id,
                  content: JSON.stringify(resp),
                };
                resolve(message);
              })
            );
          }
          const toolCallResults = await Promise.all(toolCallPromises);

          // Yield for each tool call result
          for (const toolCallResult of toolCallResults) {
            this.messages.push(toolCallResult);
            yield {
              type: "tool_call_result",
              endOfTurn: true,
              role: "tool",
              content: toolCallResult,
            };
          }
          // Infer again with a new request
          break;
        }

        // This means AI finished its answer
        if (
          delta.finish_reason === "stop" ||
          delta.finish_reason === "length" ||
          delta.finish_reason === "error"
        ) {
          yield {
            type: "output_text",
            endOfTurn: true,
            role: "assistant",
            content: (delta.message as AIOutputTextMessage).content,
          };
          // Finish this AsyncGenerator
          return;
        }
      }
    }
  }

  async deinitialize(): Promise<void> {
    if (!this.valid) return;
    const result = await this.runtime.delete(this.modelInfo.componentName);
    if (!result) throw Error(`model cleanup failed`);
    this.valid = false;
  }
}

/** function createAssistant */
export async function createAgent(
  runtime: Runtime,
  args: {
    model: TVMModelAttrs | OpenAIModelAttrs;
    tools?: Tool[];
    systemMessage?: string;
  }
): Promise<Agent> {
  const ctx = new Agent(runtime, args);
  await ctx.initialize();
  return ctx;
}
