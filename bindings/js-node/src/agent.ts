import * as MCPClientStdio from "@modelcontextprotocol/sdk/client/stdio.js";
import boxen from "boxen";
import chalk from "chalk";
import { search } from "jmespath";
import sharp from "sharp";

import MCPServer from "./mcp";
import { AiloyModel } from "./models";
import { Runtime, generateUUID } from "./runtime";
import { sharpImageToBase64 } from "./utils/image";

/** Types for internal data structures */

export class TextContent {
  constructor(
    public type: "text",
    public text: string
  ) {}
}

export class ImageContent {
  constructor(
    public type: "image_url",
    public image_url: {
      url: string;
    }
  ) {}

  static fromUrl(url: string) {
    return new ImageContent("image_url", { url });
  }

  static async fromSharp(image: sharp.Sharp) {
    return new ImageContent("image_url", {
      url: await sharpImageToBase64(image),
    });
  }
}

export class AudioContent {
  constructor(
    public type: "input_audio",
    public input_audio: {
      data: string;
      format: "mp3" | "wav";
    }
  ) {}

  static async fromBytes(data: Buffer, format: "mp3" | "wav") {
    return new AudioContent("input_audio", {
      data: data.toString("base64"),
      format,
    });
  }
}

export interface FunctionData {
  type: "function";
  id?: string;
  function: {
    name: string;
    arguments: any;
  };
}

export interface SystemMessage {
  role: "system";
  content: string | Array<TextContent>;
}

export interface UserMessage {
  role: "user";
  content: string | Array<TextContent | ImageContent | AudioContent>;
}

export interface AssistantMessage {
  role: "assistant";
  content?: string | Array<TextContent>;
  name?: string;
  tool_calls?: Array<FunctionData>;

  // Non-OpenAI fields
  reasoning?: Array<{ type: "text"; text: string }>;
}

export interface ToolMessage {
  role: "tool";
  content: Array<{ type: "text"; text: string }>;
  tool_call_id?: string;
}

export type Message =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolMessage;

export interface MessageOutput {
  message: AssistantMessage;
  finish_reason:
    | "stop"
    | "tool_calls"
    | "invalid_tool_call"
    | "length"
    | "error"
    | undefined;
}

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
    id?: string;
    function: { name: string; arguments: any };
  };
}
export interface AgentResponseToolResult {
  type: "tool_call_result";
  role: "tool";
  isTypeSwitched: boolean;
  content: ToolMessage;
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

export interface Tool {
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

  private mcpServers: MCPServer[];

  private messages: Message[];
  private systemMessage?: string;

  constructor(
    /** The runtime environment associated with the agent */
    runtime: Runtime,
    args?: {
      /** Optional system message to set the initial assistant context */
      systemMessage?: string;
    }
  ) {
    this.runtime = runtime;

    // Initialize component state
    this.componentState = {
      name: generateUUID(),
      valid: false,
    };

    // Initialize messages
    this.messages = [];

    // Initialize system message
    this.systemMessage = args?.systemMessage;

    // Initialize tools
    this.tools = [];

    // Initialize mcpServers
    this.mcpServers = [];
  }

  /**
   * Defines the LLM components to the runtime.
   * This must be called before using any other method in the class. If already defined, this is a no-op.
   */
  async define(
    /** The name of the LLM model to use in this instance */
    model: AiloyModel
  ): Promise<void> {
    // Skip if the component already exists
    if (this.componentState.valid) return;

    if (!this.runtime.is_alive()) throw Error(`Runtime is currently stopped.`);

    // Set default system message if not given; still can be undefined
    this.systemMessage =
      this.systemMessage ??
      ("defaultSystemMessage" in model
        ? model.defaultSystemMessage()
        : undefined);
    this.clearMessages();

    // Call runtime to define componenets
    const result = await this.runtime.define(
      model.componentType,
      this.componentState.name,
      model.toAttrs()
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

    if (!this.runtime.is_alive()) {
      const result = await this.runtime.delete(this.componentState.name);
      if (!result) throw Error(`component delete failed`);
    }

    // Clear messages
    this.clearMessages();

    // Cleanup MCP servers
    for (const mcpServer of this.mcpServers) {
      await mcpServer.cleanup();
    }
    this.mcpServers = [];

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

  /** Create a MCP server and register its tools to agent. */
  async addToolsFromMcpServer(
    /** The unique name of the MCP server. If there's already a MCP server with the same name, it throws Error. */
    name: string,
    /** Parameters for connecting to the MCP stdio server */
    params: MCPClientStdio.StdioServerParameters,
    options?: {
      /** Optional list of tool names to add. If not specified, all tools are added. */
      toolsToAdd?: Array<string>;
    }
  ) {
    if (this.mcpServers.some((s) => s.name === name)) {
      throw Error(`MCP server with name "${name}" is already registered`);
    }

    // Create and register MCP server
    const mcpServer = new MCPServer(name, params);
    await mcpServer.start();
    this.mcpServers.push(mcpServer);

    // Register tools
    const tools = await mcpServer.listTools();
    for (const tool of tools) {
      // If `toolsToAdd` options is provided and this tool name does not belong to them, ignore it
      if (
        options?.toolsToAdd !== undefined &&
        !options?.toolsToAdd.includes(tool.name)
      )
        continue;

      const desc: ToolDescription = {
        name: `${name}-${tool.name}`,
        description: tool.description || "",
        parameters: tool.inputSchema as ToolDescription["parameters"],
      };
      this.addTool({
        desc,
        call: async (inputs) => await mcpServer.callTool(tool, inputs),
      });
    }
  }

  /** Removes the MCP server and its tools from the agent, with terminating the MCP server process. */
  async removeMcpServer(
    /** The unique name of the MCP server. If there's no MCP server matches the name, it throws Error. */
    name: string
  ) {
    if (this.mcpServers.every((s) => s.name !== name)) {
      throw Error(`MCP server with name "${name}" does not exist`);
    }

    // Remove the MCP server
    const idx = this.mcpServers.findIndex((s) => s.name === name)!;
    await this.mcpServers[idx].cleanup();
    this.mcpServers.splice(idx, 1);

    // Remove tools registered from the MCP server
    this.tools = this.tools.filter((t) => !t.desc.name.startsWith(`${name}-`));
  }

  getAvailableTools(): Array<ToolDescription> {
    return this.tools.map((tool) => tool.desc);
  }

  async *query(
    /** The user message to send to the model */
    message:
      | string
      | Array<string | sharp.Sharp | TextContent | ImageContent | AudioContent>,
    options?: {
      /** If True, enables reasoning capabilities (default: False) */
      reasoning?: boolean;
    }
  ): AsyncGenerator<AgentResponse> {
    if (!this.componentState.valid)
      throw Error(`Agent is not valid. Create one or define newly.`);

    if (!this.runtime.is_alive()) throw Error(`Runtime is currently stopped.`);

    if (typeof message === "string") {
      this.messages.push({
        role: "user",
        content: [{ type: "text", text: message }],
      });
    } else {
      if (message.length === 0) {
        throw Error("Message is empty");
      }

      let contents: Array<TextContent | ImageContent | AudioContent> = [];
      for (const content of message) {
        if (typeof content === "string") {
          contents.push({ type: "text", text: content });
        } else if (
          ((obj: any): obj is sharp.Sharp => obj instanceof sharp)(content)
        ) {
          contents.push(await ImageContent.fromSharp(content));
        } else {
          contents.push(content);
        }
      }

      this.messages.push({ role: "user", content: contents });
    }

    let prevRespType: string | null = null;

    while (true) {
      let assistantReasoning: Array<TextContent> | undefined = undefined;
      let assistantContent: Array<TextContent> | undefined = undefined;
      let assistantToolCalls: Array<FunctionData> | undefined = undefined;
      let finishReason = "";

      for await (const result of this.runtime.callIterMethod(
        this.componentState.name,
        "infer",
        {
          messages: this.messages,
          tools: this.tools.map((v) => {
            return { type: "function", function: v.desc };
          }),
          reasoning: options?.reasoning,
        }
      ) as AsyncIterable<MessageOutput>) {
        if (result.message.reasoning) {
          for (const reasoningData of result.message.reasoning) {
            if (assistantReasoning === undefined)
              assistantReasoning = [reasoningData];
            else assistantReasoning[0].text += reasoningData.text;
            const resp: AgentResponseText = {
              type: "reasoning",
              role: "assistant",
              isTypeSwitched: prevRespType !== "reasoning",
              content: reasoningData.text,
            };
            prevRespType = resp.type;
            yield resp;
          }
        }
        if (result.message.content !== undefined) {
          // Canonicalize message content to the array of TextContent
          if (typeof result.message.content == "string") {
            result.message.content = [
              { type: "text", text: result.message.content },
            ];
          }
          for (const contentData of result.message.content) {
            if (assistantContent === undefined)
              assistantContent = [contentData];
            else assistantContent[0].text += contentData.text;
            const resp: AgentResponseText = {
              type: "output_text",
              role: "assistant",
              isTypeSwitched: prevRespType !== "output_text",
              content: contentData.text,
            };
            prevRespType = resp.type;
            yield resp;
          }
        }
        if (result.message.tool_calls) {
          for (const tool_call_data of result.message.tool_calls) {
            if (assistantToolCalls === undefined)
              assistantToolCalls = [tool_call_data];
            else assistantToolCalls.push(tool_call_data);
            const resp: AgentResponseToolCall = {
              type: "tool_call",
              role: "assistant",
              isTypeSwitched: true,
              content: tool_call_data,
            };
            prevRespType = resp.type;
            yield resp;
          }
        }

        if (result.finish_reason) {
          finishReason = result.finish_reason;
          break;
        }
      }
      // Append output
      this.messages.push({
        role: "assistant",
        content: assistantContent,
        reasoning: assistantReasoning,
        tool_calls: assistantToolCalls,
      });

      // Call tools in parallel
      if (finishReason == "tool_calls") {
        let toolCallPromises: Array<Promise<ToolMessage>> = [];
        for (const toolCall of assistantToolCalls ?? []) {
          toolCallPromises.push(
            new Promise(async (resolve, reject) => {
              const tool_ = this.tools.find(
                (v) => v.desc.name == toolCall.function.name
              );
              if (!tool_) {
                reject("Internal exception");
                return;
              }
              const toolResult = await tool_.call(toolCall.function.arguments);
              const message: ToolMessage = {
                role: "tool",
                content: [{ type: "text", text: JSON.stringify(toolResult) }],
                tool_call_id: toolCall.id,
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
            isTypeSwitched: true,
            content: toolCallResult,
          };
          prevRespType = resp.type;
          yield resp;
        }
        // Infer again if tool calls happened
        continue;
      }

      // Finish this generator
      break;
    }
  }

  /**
   * Get the current conversation history.
   * Each item in the list represents a message from either the user or the assistant.
   */
  getMessages() {
    return this.messages;
  }

  /**
   * Clear the history of conversation messages.
   */
  clearMessages() {
    this.messages = [];
    if (this.systemMessage !== undefined)
      this.messages.push({
        role: "system",
        content: [{ type: "text", text: this.systemMessage }],
      });
  }

  /**
   * Get the list of registered tools.
   */
  getTools() {
    return this.tools;
  }

  /**
   * Clear the registered tools.
   */
  clearTools() {
    this.tools = [];
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
    let title =
      chalk.magenta("Tool Call") +
      ": " +
      chalk.bold(resp.content.function.name);
    if (resp.content.id !== undefined && resp.content.id.length > 0) {
      title += ` (${resp.content.id})`;
    }
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
    let title = chalk.green("Tool Result");
    if (
      resp.content.tool_call_id !== undefined &&
      resp.content.tool_call_id.length > 0
    ) {
      title += ` (${resp.content.tool_call_id})`;
    }

    let content: string;
    try {
      // Try to parse as json
      content = JSON.stringify(
        JSON.parse(resp.content.content[0].text),
        null,
        2
      );
    } catch (e) {
      // Use original content if not json deserializable
      content = resp.content.content[0].text;
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
  model: AiloyModel,
  args?: {
    /** Optional system message to set the initial assistant context */
    systemMessage?: string;
  }
): Promise<Agent> {
  // Call constructor
  const agent = new Agent(runtime, args);

  await agent.define(model);

  // Return created agent
  return agent;
}
