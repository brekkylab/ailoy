import * as MCPClient from "@modelcontextprotocol/sdk/client/index.js";
import * as MCPClientStdio from "@modelcontextprotocol/sdk/client/stdio.js";
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

export type TVMModelName =
  | "qwen3-8b"
  | "qwen3-4b"
  | "qwen3-1.7b"
  | "qwen3-0.6b";

export type OpenAIModelName = "gpt-4o";

export type ModelName = TVMModelName | OpenAIModelName;

interface ModelDescription {
  modelId: string;
  componentType: string;
  defaultSystemMessage?: string;
}

const modelDescriptions: Record<ModelName, ModelDescription> = {
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

export interface AgentResponseText {
  type: "output_text" | "reasoning";
  role: "assistant";
  endOfTurn: boolean;
  content: string;
}
export interface AgentResponseToolCall {
  type: "tool_call";
  role: "assistant";
  endOfTurn: boolean;
  content: {
    id: string;
    function: { name: string; arguments: any };
  };
}
export interface AgentResponseToolResult {
  type: "tool_call_result";
  role: "tool";
  endOfTurn: boolean;
  content: {
    name: string;
    tool_call_id: string;
    content: string;
  };
}
export interface AgentResponseError {
  type: "error";
  role: "assistant";
  endOfTurn: true;
  content: string;
}
export type AgentResponse =
  | AgentResponseText
  | AgentResponseToolCall
  | AgentResponseToolResult
  | AgentResponseError;

/** Types and functions related to Tools */

interface Tool {
  desc: ToolDescription;
  call: (runtime: Runtime, input: any) => Promise<any>;
}

export interface ToolDescription {
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

export interface ToolDefinitionUniversal {
  type: "universal";
  description: ToolDescription;
  behavior: {
    outputPath?: string;
  };
}

export interface ToolDefinitionRESTAPI {
  type: "restapi";
  description: ToolDescription;
  behavior: {
    baseURL: string;
    method: "GET" | "POST" | "PUT" | "DELETE";
    headers: { [key: string]: string };
    body?: string;
    outputPath?: string;
  };
}

export type ToolDefinition = ToolDefinitionUniversal | ToolDefinitionRESTAPI;

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

  private componentState: {
    name: string;
    valid: boolean;
  };

  private tools: Tool[];

  private messages: Message[];

  constructor(runtime: Runtime, systemMessage?: string) {
    this.runtime = runtime;

    // Initialize component state
    this.componentState = {
      name: generateUUID(),
      valid: false,
    };

    // Initialize messages
    this.messages = [];

    // Use system message provided from arguments first, otherwise use default system message for the model.
    this.messages = [];
    if (systemMessage !== undefined)
      this.messages.push({ role: "system", content: systemMessage });

    // Initialize tools
    this.tools = [];
  }

  async define(
    modelName: ModelName,
    attrs: Record<string, any>
  ): Promise<void> {
    // Skip if the component already exists
    if (this.componentState.valid) return;

    const modelDesc = modelDescriptions[modelName];

    // Add model name into attrs
    if (!attrs.model) attrs.model = modelDesc.modelId;

    // Set default system message
    if (this.messages.length == 0 && modelDesc.defaultSystemMessage)
      this.messages.push({
        role: "system",
        content: modelDesc.defaultSystemMessage,
      });

    // Call runtime to define componenets
    const result = await this.runtime.define(
      modelDesc.componentType,
      this.componentState.name,
      attrs
    );
    if (!result) throw Error(`component define failed`);

    // Mark component as defined
    this.componentState.valid = true;
  }

  async delete(): Promise<void> {
    // Skip if the component not exists
    if (!this.componentState.valid) return;

    const result = await this.runtime.delete(this.componentState.name);
    if (!result) throw Error(`component delete failed`);

    // Clear messages
    if (this.messages.length > 0 && this.messages[0].role === "system")
      this.messages = [this.messages[0]];
    else this.messages = [];

    // Mark component as deleted
    this.componentState.valid = false;
  }

  addTool(tool: Tool): boolean {
    if (this.tools.find((t) => t.desc.name == tool.desc.name)) return false;
    this.tools.push(tool);
    return true;
  }

  addJSFunctionTool(desc: ToolDescription, f: (input: any) => any): boolean {
    return this.addTool({ desc, call: f });
  }

  addUniversalTool(tool: ToolDefinitionUniversal): boolean {
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
    tool: ToolDefinitionRESTAPI,
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

  async addMcpTool(
    params: MCPClientStdio.StdioServerParameters,
    tool: Awaited<ReturnType<MCPClient.Client["listTools"]>>["tools"][number]
  ) {
    const call = async (_: Runtime, inputs: any) => {
      const transport = new MCPClientStdio.StdioClientTransport(params);
      const client = new MCPClient.Client({
        name: "dummy-client",
        version: "dummy-version",
      });
      await client.connect(transport);

      const { content } = await client.callTool({
        name: tool.name,
        arguments: inputs,
      });
      return content;
    };
    const desc: ToolDescription = {
      name: tool.name,
      description: tool.description || "",
      parameters: tool.inputSchema as ToolDescription["parameters"],
    };
    return this.addTool({ desc, call });
  }

  async addToolsFromMcpServer(
    params: MCPClientStdio.StdioServerParameters,
    options?: {
      toolsToAdd?: Array<string>;
    }
  ) {
    const transport = new MCPClientStdio.StdioClientTransport(params);
    const client = new MCPClient.Client({
      name: "dummy-client",
      version: "dummy-version",
    });
    await client.connect(transport);

    const { tools } = await client.listTools();
    for (const tool of tools) {
      // If `toolsToAdd` options is provided and this tool name does not belong to them, ignore it
      if (
        options?.toolsToAdd !== undefined &&
        !options?.toolsToAdd.includes(tool.name)
      )
        continue;
      await this.addMcpTool(params, tool);
    }
  }

  getAvailableTools(): Array<ToolDescription> {
    return this.tools.map((tool) => tool.desc);
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

  async *run(
    message: string,
    options?: {
      enableReasoning?: boolean;
      ignoreReasoningMessages?: boolean;
    }
  ): AsyncGenerator<AgentResponse> {
    this.messages.push({ role: "user", content: message });

    while (true) {
      for await (const resp of this.runtime.callIterMethod(
        this.componentState.name,
        "infer",
        {
          messages: this.messages,
          tools: this.tools.map((v) => {
            return { type: "function", function: v.desc };
          }),
          enable_reasoning: options?.enableReasoning,
          ignore_reasoning_messages: options?.ignoreReasoningMessages,
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
}

export async function defineAgent(
  runtime: Runtime,
  modelName: ModelName,
  args?: {
    systemMessage?: string;
    apiKey?: string;
  }
): Promise<Agent> {
  const args_ = args || {};

  // Call constructor
  const agent = new Agent(runtime, args_.systemMessage);

  // Attribute input for call `rt.define`
  let attrs: Record<string, any> = {};
  if (args_.apiKey) attrs["api_key"] = args_.apiKey;

  await agent.define(modelName, attrs);

  // Return created agent
  return agent;
}
