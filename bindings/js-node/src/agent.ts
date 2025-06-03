import * as MCPClient from "@modelcontextprotocol/sdk/client/index.js";
import * as MCPClientStdio from "@modelcontextprotocol/sdk/client/stdio.js";
import boxen from "boxen";
import chalk from "chalk";
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
  | "Qwen/Qwen3-8B"
  | "Qwen/Qwen3-4B"
  | "Qwen/Qwen3-1.7B"
  | "Qwen/Qwen3-0.6B";

export type OpenAIModelName = "gpt-4o";

export type ModelName = TVMModelName | OpenAIModelName;

interface ModelDescription {
  modelId: string;
  componentType: string;
  defaultSystemMessage?: string;
}

const modelDescriptions: Record<ModelName, ModelDescription> = {
  "Qwen/Qwen3-8B": {
    modelId: "Qwen/Qwen3-8B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "Qwen/Qwen3-4B": {
    modelId: "Qwen/Qwen3-4B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "Qwen/Qwen3-1.7B": {
    modelId: "Qwen/Qwen3-1.7B",
    componentType: "tvm_language_model",
    defaultSystemMessage:
      "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  },
  "Qwen/Qwen3-0.6B": {
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
  isTypeSwitched: boolean;
  content: string;
}
export interface AgentResponseToolCall {
  type: "tool_call";
  role: "assistant";
  isTypeSwitched: boolean;
  content: {
    id: string;
    function: { name: string; arguments: any };
  };
}
export interface AgentResponseToolResult {
  type: "tool_call_result";
  role: "tool";
  isTypeSwitched: boolean;
  content: {
    name: string;
    tool_call_id: string;
    content: string;
  };
}
export interface AgentResponseError {
  type: "error";
  role: "assistant";
  isTypeSwitched: boolean;
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
  call: (input: any) => Promise<any>;
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

export interface ToolDefinitionBuiltin {
  type: "builtin";
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

export type ToolDefinition = ToolDefinitionBuiltin | ToolDefinitionRESTAPI;

export type ToolAuthenticator = (request: {
  url: string;
  headers: { [k: string]: string };
  [k: string]: any;
}) => {
  url: string;
  headers: { [k: string]: string };
  [k: string]: any;
};

export function bearerAutenticator(
  token: string,
  bearerFormat: string = "Bearer"
): ToolAuthenticator {
  return (request: {
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
  };
}

/**
 * The `Agent` class provides a high-level interface for interacting with large language models (LLMs) in Ailoy.
 * It abstracts the underlying runtime and VM logic, allowing users to easily send queries and receive streaming
 * responses.
 * Agents can be extended with external tools or APIs to provide real-time or domain-specific knowledge, enabling
 * more powerful and context-aware interactions.
 */
export class Agent {
  private runtime: Runtime;

  private componentState: {
    name: string;
    valid: boolean;
  };

  private tools: Tool[];

  private messages: Message[];

  constructor(
    /** The runtime environment associated with the agent */
    runtime: Runtime,
    /** Optional system message to set the initial assistant context */
    systemMessage?: string
  ) {
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

  /**
   * Defines the LLM components to the runtime.
   * This must be called before using any other method in the class. If already defined, this is a no-op.
   */
  async define(
    /** The name of the LLM model to use in this instance */
    modelName: ModelName,
    /** `Additional input used as an attribute in the define call of Runtime */
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

  /**
   * Delete resources from the runtime.
   * This should be called when the VectorStore is no longer needed. If already deleted, this is a no-op.
   */
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

  /** Adds a custom tool to the agent */
  addTool(
    /** Tool instance to be added */
    tool: Tool
  ): boolean {
    if (this.tools.find((t) => t.desc.name == tool.desc.name)) return false;
    this.tools.push(tool);
    return true;
  }

  /** Adds a Javascript function as a tool using callable */
  addJSFunctionTool(
    /** Function will be called when the tool invocation occured */
    f: (input: any) => any,
    /** Tool descriotion */
    desc: ToolDescription
  ): boolean {
    return this.addTool({ desc, call: f });
  }

  addBuiltinTool(
    /** The built in tool definition */
    tool: ToolDefinitionBuiltin
  ): boolean {
    const call = async (inputs: any) => {
      // Validation
      const required = tool.description.parameters.required || [];
      const missing = required.filter((name) => !(name in inputs));
      if (missing.length != 0)
        throw Error("some parameters are required but not exist: " + missing);

      // Call
      let output = await this.runtime.call(tool.description.name, inputs);

      // Parse output path
      if (tool.behavior.outputPath)
        output = search(output, tool.behavior.outputPath);

      // Return
      return output;
    };
    return this.addTool({ desc: tool.description, call });
  }

  /** Adds a REST API tool that performs external HTTP requests */
  addRESTAPITool(
    /** REST API tool definition */
    tool: ToolDefinitionRESTAPI,
    /** Optional authenticator to inject into the request */
    auth?: ToolAuthenticator
  ): boolean {
    const call = async (inputs: any) => {
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
      const request = auth ? auth(requestNoAuth) : requestNoAuth;

      // Call
      let output: any;
      const resp = await this.runtime.call("http_request", request);
      // @jhlee: How to parse it?
      output = JSON.parse(resp.body);

      // Parse output path
      if (outputPath) output = search(output, outputPath);

      // Return
      return output;
    };
    return this.addTool({ desc: tool.description, call });
  }

  /** Loads tools from a predefined JSON preset file */
  addToolsFromPreset(
    /** Name of the tool preset */
    presetName: string,
    args?: {
      /** Optional authenticator to inject into the request */
      authenticator?: ToolAuthenticator;
    }
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
      } else if (tool.type == "builtin") {
        const result = this.addBuiltinTool(tool);
        if (!result) return false;
      }
    }
    return true;
  }

  /** Adds a tool from an MCP (Model Context Protocol) server */
  async addMcpTool(
    /** Parameters for connecting to the MCP stdio server */
    params: MCPClientStdio.StdioServerParameters,
    /** Tool metadata as defined by MCP */
    tool: Awaited<ReturnType<MCPClient.Client["listTools"]>>["tools"][number]
  ) {
    const call = async (inputs: any) => {
      const transport = new MCPClientStdio.StdioClientTransport(params);
      const client = new MCPClient.Client({
        name: "dummy-client",
        version: "dummy-version",
      });
      await client.connect(transport);

      const result = await client.callTool({
        name: tool.name,
        arguments: inputs,
      });
      const content = result.content as Array<any>;
      const parsedContent = content.map((item) => {
        if (item.type === "text") {
          // Text Content
          return item.text;
        } else if (item.type === "image") {
          // Image Content
          return item.data;
        } else if (item.type === "resource") {
          // Resource Content
          if (item.resource.text !== undefined) {
            // Text Resource
            return item.resource.text;
          } else {
            // Blob Resource
            return item.resource.blob;
          }
        }
      });
      await client.close();

      return parsedContent;
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
    await client.close();
  }

  getAvailableTools(): Array<ToolDescription> {
    return this.tools.map((tool) => tool.desc);
  }

  async *query(
    /** The user message to send to the model */
    message: string,
    options?: {
      /** If True, enables reasoning capabilities (default: True) */
      enableReasoning?: boolean;
      /** If True, reasoning steps are not included in the response stream */
      ignoreReasoningMessages?: boolean;
    }
  ): AsyncGenerator<AgentResponse> {
    this.messages.push({ role: "user", content: message });

    let prevRespType: string | null = null;
    let isToolCalled = false;

    while (true) {
      let reasoningMessage = "";
      let outputTextMessage = "";

      const _pushAssistantTextMessage = () => {
        // if output_text is empty, do nothing
        if (outputTextMessage.trim() === "") return;

        // construct assistant message content considering reasoning message
        let assistantMessageContent = outputTextMessage.trim();
        if (
          options?.enableReasoning &&
          !options.ignoreReasoningMessages &&
          reasoningMessage.trim() !== ""
        )
          // TODO: consider other kinds of reasoning part distinguisher
          assistantMessageContent =
            `<think>\n${reasoningMessage.trim()}\n</think>\n\n` +
            assistantMessageContent;

        // push constructed message content into messages
        this.messages.push({
          role: "assistant",
          content: assistantMessageContent,
        } as AIOutputTextMessage);

        reasoningMessage = "";
        outputTextMessage = "";
      };

      for await (const result of this.runtime.callIterMethod(
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
        const delta: MessageDelta = result;

        // This means AI requested tool calls
        if (delta.finish_reason === "tool_calls") {
          isToolCalled = true;
          _pushAssistantTextMessage();

          const toolCallMessage = delta.message as AIToolCallMessage;
          // Add tool call back to messages
          this.messages.push(toolCallMessage);

          // Yield for each tool call
          for (const toolCall of toolCallMessage.tool_calls) {
            const resp: AgentResponseToolCall = {
              type: "tool_call",
              role: "assistant",
              isTypeSwitched: prevRespType !== "tool_call",
              content: toolCall,
            };
            prevRespType = resp.type;
            yield resp;
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
                const resp = await tool_.call(toolCall.function.arguments);
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
            const resp: AgentResponseToolResult = {
              type: "tool_call_result",
              role: "tool",
              isTypeSwitched: prevRespType !== "tool_call_result",
              content: toolCallResult,
            };
            prevRespType = resp.type;
            yield resp;
          }

          continue;
        }

        let message = delta.message as AIOutputTextMessage;

        // Accumulate text messages to append in batch
        if (message.reasoning) reasoningMessage += message.content;
        else outputTextMessage += message.content;

        const resp: AgentResponseText = {
          type: message.reasoning ? "reasoning" : "output_text",
          role: "assistant",
          isTypeSwitched: false,
          content: message.content,
        };
        resp.isTypeSwitched = prevRespType !== resp.type;
        prevRespType = resp.type;
        yield resp;

        // This means AI finished its answer
        if (
          delta.finish_reason === "stop" ||
          delta.finish_reason === "length" ||
          delta.finish_reason === "error"
        ) {
          _pushAssistantTextMessage();
          // Finish this infer
          break;
        }
      }

      // Infer again if tool calls happened
      if (isToolCalled) {
        isToolCalled = false;
        continue;
      }

      // Finish this generator
      return;
    }
  }

  private _printResponseText(resp: AgentResponseText) {
    if (resp.isTypeSwitched) {
      process.stdout.write("\n");
    }
    const content =
      resp.type === "reasoning" ? chalk.yellow(resp.content) : resp.content;
    process.stdout.write(content);
  }

  private _printResponseToolCall(resp: AgentResponseToolCall) {
    const title =
      chalk.magenta("Tool Call") +
      ": " +
      chalk.bold(resp.content.function.name) +
      ` (${resp.content.id})`;
    const content = JSON.stringify(resp.content.function.arguments, null, 2);
    const box = boxen(content, {
      title,
      titleAlignment: "left",
      padding: {
        left: 1,
        right: 1,
        top: 0,
        bottom: 0,
      },
    });
    console.log(box);
  }

  private _printResponseToolResult(resp: AgentResponseToolResult) {
    const title =
      chalk.green("Tool Result") +
      ": " +
      chalk.bold(resp.content.name) +
      ` (${resp.content.tool_call_id})`;

    let content;
    try {
      // Try to parse as json
      content = JSON.stringify(JSON.parse(resp.content.content), null, 2);
    } catch (e) {
      // Use original content if not json deserializable
      content = resp.content.content;
    }
    // Truncate long contents
    if (content.length > 500) {
      content = content.slice(0, 500) + "...(truncated)";
    }

    const box = boxen(content, {
      title,
      titleAlignment: "left",
      padding: {
        left: 1,
        right: 1,
        top: 0,
        bottom: 0,
      },
    });
    console.log(box);
  }

  private _printResponseError(resp: AgentResponseError) {
    const title = chalk.red.bold("Error");
    const box = boxen(resp.content, {
      title,
      titleAlignment: "left",
      padding: {
        left: 1,
        right: 1,
        top: 0,
        bottom: 0,
      },
    });
    console.log(box);
  }

  /** Prints agent's responses in a pretty format */
  print(
    /** agent's response yielded from `query()` */
    resp: AgentResponse
  ) {
    if (resp.type === "output_text" || resp.type === "reasoning") {
      this._printResponseText(resp);
    } else if (resp.type === "tool_call") {
      this._printResponseToolCall(resp);
    } else if (resp.type === "tool_call_result") {
      this._printResponseToolResult(resp);
    } else if (resp.type === "error") {
      this._printResponseError(resp);
    }
  }
}

/** Define a new agent */
export async function defineAgent(
  /** The runtime environment associated with the agent */
  runtime: Runtime,
  /** The name of the LLM model to use in this instance */
  modelName: ModelName,
  args?: {
    /** Optional system message to set the initial assistant context */
    systemMessage?: string;
    /** Optional device id to set the device id to run LLM model */
    device?: number;
    /** A parameter for API key usage.
     * This field is ignored if the model does not require authentication. */
    apiKey?: string;
  }
): Promise<Agent> {
  const args_ = args || {};

  // Call constructor
  const agent = new Agent(runtime, args_.systemMessage);

  // Attribute input for call `rt.define`
  let attrs: Record<string, any> = {};
  if (args_.device) attrs["device"] = args_.device;
  if (args_.apiKey) attrs["api_key"] = args_.apiKey;

  await agent.define(modelName, attrs);

  // Return created agent
  return agent;
}
