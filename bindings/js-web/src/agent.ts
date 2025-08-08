import { search } from "jmespath";
import { Image } from "wasm-vips";

import { MCPClient, MCPClientTransport, MCPClientStartOptions } from "./mcp";
import { _LocalModel, AiloyModel } from "./models";
import { Runtime } from "./runtime";
import { uint8ArrayToBase64 } from "./utils/base64";
import { isVipsImage, vipsImageToBase64 } from "./utils/image";

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

  static fromVips(image: Image) {
    return new ImageContent("image_url", {
      url: vipsImageToBase64(image),
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

  static async fromBytes(
    data: Uint8Array<ArrayBufferLike>,
    format: "mp3" | "wav"
  ) {
    return new AudioContent("input_audio", {
      data: uint8ArrayToBase64(data),
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

export type JsonSchemaTypes =
  | "string"
  | "integer"
  | "number"
  | "boolean"
  | "object"
  | "array"
  | "null";

export interface ToolDescription {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: {
      [key: string]: {
        type: JsonSchemaTypes | JsonSchemaTypes[];
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
  private messages: Message[];
  private systemMessage?: string;
  private tools: Tool[];
  private mcpClients: Map<string, MCPClient>;

  #initialized: boolean = false;
  private runtime?: Runtime;
  private model?: AiloyModel;

  constructor() {
    // Initialize messages
    this.messages = [];

    // Initialize tools
    this.tools = [];

    // Initialize mcpClients
    this.mcpClients = new Map();
  }

  /**
   * Defines the LLM components to the runtime.
   * This must be called before using any other method in the class. If already defined, this is a no-op.
   */
  async define(
    /** The runtime environment associated with the agent */
    runtime: Runtime,
    /** The LLM model to use in this agent */
    model: AiloyModel,
    args?: {
      /** Optional system message to set the initial assistant context */
      systemMessage?: string;
    }
  ): Promise<void> {
    if (this.#initialized) return;

    if (!runtime.isAlive()) throw Error(`Runtime is not alive.`);
    this.runtime = runtime;

    this.model = model;
    await this.model.init({ runtime: this.runtime });

    // Set default system message if not given; still can be undefined
    this.systemMessage = args?.systemMessage ?? model.defaultSystemMessage();
    this.clearMessages();

    this.#initialized = true;
  }

  /**
   * Delete resources in the agent.
   * If the agent is not in an initialized state, this is a no-op.
   */
  async delete(): Promise<void> {
    // Skip if the agent is not initialized
    if (!this.#initialized) return;

    this.runtime = undefined;

    await this.model!.dispose();
    this.model = undefined;

    // Clear messages
    this.systemMessage = undefined;
    this.clearMessages();

    // Close MCP clients
    for (const [name, mcpClient] of this.mcpClients) {
      await mcpClient.cleanup();
    }
    this.mcpClients.clear();

    this.#initialized = false;
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
      let output = await this.runtime!.call(tool.description.name, inputs);

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
      const resp = await this.runtime!.call("http_request", request);
      output = resp.body;

      // parse as JSON if "accept" header is "application/json"
      if (tool.behavior.headers.accept === "application/json") {
        const decoder = new TextDecoder();
        output = JSON.parse(decoder.decode(output));
      }

      // Parse output path
      if (outputPath) output = search(output, outputPath);

      // Return
      return output;
    };
    return this.addTool({ desc: tool.description, call });
  }

  /** Loads tools from a predefined JSON preset file */
  async addToolsFromPreset(
    /** Name of the tool preset */
    presetName: string,
    args?: {
      /** Optional authenticator to inject into the request */
      authenticator?: ToolAuthenticator;
    }
  ): Promise<boolean> {
    const presetJsonUrl = new URL(
      `./presets/tools/${presetName}.json`,
      import.meta.url
    );
    const presetJsonResp = await fetch(presetJsonUrl);
    if (!presetJsonResp.ok) {
      throw Error(`Preset "${presetName}" does not exist`);
    }
    const presetJson = await presetJsonResp.json();

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

  /** Register a MCP client and add its tools to agent. */
  async addToolsFromMcpClient(
    /** The unique name of the MCP client. If there's already a MCP client with the same name, it throws Error. */
    name: string,
    /** MCP client transport to use for client connection */
    transport: MCPClientTransport,
    options?: MCPClientStartOptions & {
      /** Optional list of tool names to add. If not specified, all tools are added. */
      toolsToAdd?: Array<string>;
    }
  ) {
    if (this.mcpClients.has(name)) {
      throw Error(`MCP client with name "${name}" is already registered`);
    }

    // Start MCP client and register to agent
    const client = new MCPClient(name);
    await client.start(transport, options);
    this.mcpClients.set(name, client);

    // Register tools
    const tools = await client.listTools();
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
        call: async (inputs) => await client.callTool(tool, inputs),
      });
    }
  }

  /** Removes the MCP client and its tools from the agent. */
  async removeMcpClient(
    /** The unique name of the MCP client. If there's no MCP client matches the name, it throws Error. */
    name: string
  ) {
    if (!this.mcpClients.has(name)) {
      throw Error(`MCP client with name "${name}" does not exist`);
    }

    // Remove the MCP client
    await this.mcpClients.get(name)?.cleanup();
    this.mcpClients.delete(name);

    // Remove tools registered from the MCP client
    this.tools = this.tools.filter((t) => !t.desc.name.startsWith(`${name}-`));
  }

  async *query(
    message:
      | string
      | Array<string | Image | TextContent | ImageContent | AudioContent>,
    options?: {
      reasoning?: boolean;
    }
  ): AsyncGenerator<AgentResponse> {
    if (!this.#initialized) throw Error(`Agent is not initialized yet.`);

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
        } else if (isVipsImage(content)) {
          contents.push(ImageContent.fromVips(content));
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

      for await (const result of await this.model!.infer({
        messages: this.messages,
        tools: this.tools,
        reasoning: options?.reasoning,
      })) {
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
}

/** Define a new agent */
export async function defineAgent(
  /** The runtime environment associated with the agent */
  runtime: Runtime,
  /** The LLM model to use in this agent */
  model: AiloyModel,
  args?: {
    /** Optional system message to set the initial assistant context */
    systemMessage?: string;
  }
): Promise<Agent> {
  const agent = new Agent();
  await agent.define(runtime, model, args);
  return agent;
}
