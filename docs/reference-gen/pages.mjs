// The reference's structure and the words the bindings' doc comments don't have.
//
// PAGES: the pages of each language, in sidebar order, the same in both, with the names each one
// documents. Every exported name must be on one, or regen stops.
//
// FILL: descriptions for what the doc comments leave out, keyed by the Python name
// (`AgentBuilder.web_search_tool`, a parameter as `AgentBuilder.web_search_tool.engines`, a field
// as `Message.role`). A doc comment, where there is one, comes first; `more` is added after it.
// Text is a string, or { python, nodejs } where the languages differ. In text, {{run_stream}} is
// a name in the page's language (`run_stream`, `runStream`), {{None}} its null.
//
// Written from the Rust source of v0.3.2; check them against the next release's when regenerating.

const g = (slug, title) => ({ slug, title });
const GUIDES = {
  agents: g('guides/agents', 'Defining an agent'),
  running: g('guides/running', 'Running an agent'),
  sandbox: g('guides/sandbox', 'Giving an agent a sandbox'),
  context: g('guides/context-manager', 'Context manager'),
  models: g('guides/model-providers', 'Model providers'),
  messages: g('guides/message-format', 'Message format'),
  tools: g('guides/tools', 'Tools'),
  mcp: g('guides/using-mcp', 'Using MCP'),
  browser: g('guides/using-browser', 'Using a browser'),
  skills: g('guides/skills', 'Skills'),
};

const RAISES = { python: 'raises', nodejs: 'throws' };

// The sidebar's groups for each language, in order. `index` is the language's Installation page;
// the sidebar puts the reference's Overview before it.
export const SECTIONS = [
  { title: 'Get started', pages: ['index', 'errors'] },
  { title: 'Agents', pages: ['agent-builder', 'agent', 'agent-run'] },
  { title: 'Models', pages: ['lang-model', 'model-providers'] },
  { title: 'Tools', pages: ['tools'] },
  { title: 'Data shapes', pages: ['messages', 'outputs', 'specs-and-options'] },
];

export const PAGES = [
  {
    slug: 'agent-builder',
    title: 'AgentBuilder',
    summary: 'Configures an agent, its model, instruction, tools, console, browser, memory and skills, and builds it.',
    guides: [GUIDES.agents, GUIDES.tools, GUIDES.sandbox, GUIDES.skills],
    items: ['AgentBuilder'],
    intro: {
      python: `\`AgentBuilder\` collects everything an agent is made of, then \`build()\` turns it into an [\`Agent\`](../agent/). Start one with the model's desc, call the methods for what the agent needs, and await \`build()\`.

Each method changes the builder in place and returns it, so calls chain. Nothing is checked or started until \`build()\`. That's when the model is built through the provider set for its org, each tool through the provider set for its name, and each skill's \`SKILL.md\` is read through the console. A missing provider raises [\`AiloyError\`](../errors/) from \`build()\`, and a \`SKILL.md\` the console can't read raises the console's own failure, such as \`virtx.ConsoleRefused\`.

A builder makes one agent. Once \`build()\` has run, every method raises \`ValueError\` ("this AgentBuilder has already been built").`,
      nodejs: `\`AgentBuilder\` collects everything an agent is made of, then \`build()\` turns it into an [\`Agent\`](../agent/). Start one with \`new AgentBuilder(model)\`, call the methods for what the agent needs, and await \`build()\`.

Each method changes the builder in place and returns it, so calls chain. Nothing is checked or started until \`build()\`. That's when the model is built through the provider set for its org, each tool through the provider set for its name, and each skill's \`SKILL.md\` is read through the console. A missing provider rejects \`build()\` with an [\`AILOY_ERROR\`](../errors/), and a \`SKILL.md\` the console can't read rejects it with the console's code, such as \`NOT_FOUND\`.

A builder makes one agent. Once \`build()\` has run, every method throws an error with code \`INVALID_ARG\` ("this AgentBuilder has already been built").`,
    },
  },
  {
    slug: 'agent',
    title: 'Agent',
    summary: 'A built agent: runs turns on a query and keeps the conversation.',
    guides: [GUIDES.running, GUIDES.agents],
    items: ['Agent'],
    intro: {
      python: `An \`Agent\` is a model, its tools and the conversation so far. Get one from \`await AgentBuilder(model)...build()\`, or from a spec with \`Agent.from_spec\`.

Each call to \`run()\` or \`run_stream()\` is one turn. The query joins the history, the model answers, the agent runs the tools it calls and hands their results back, and the model goes on until it answers without calling a tool. The history carries over, so each turn sees everything before it. If the model fails before its first message of a turn, the query is taken back out of the history, so the agent can be run again.

An agent runs one turn at a time. While a turn is being iterated, \`history\` and \`start_browser()\` raise [\`AiloyError\`](../errors/) ("this agent is in the middle of a turn"). After \`close()\` they raise "this agent has been closed".

\`Agent\` is an async context manager: \`async with await builder.build() as agent:\` closes it on the way out.`,
      nodejs: `An \`Agent\` is a model, its tools and the conversation so far. Get one from \`await new AgentBuilder(model)...build()\`, or from a spec with \`Agent.fromSpec\`.

Each call to \`run()\` or \`runStream()\` is one turn. The query joins the history, the model answers, the agent runs the tools it calls and hands their results back, and the model goes on until it answers without calling a tool. The history carries over, so each turn sees everything before it. If the model fails before its first message of a turn, the query is taken back out of the history, so the agent can be run again.

An agent runs one turn at a time. While a turn is being iterated, \`history\` and \`startBrowser()\` throw an [\`AILOY_ERROR\`](../errors/) ("this agent is in the middle of a turn"). After \`close()\` they throw "this agent has been closed".`,
    },
  },
  {
    slug: 'agent-run',
    title: 'AgentRun',
    summary: 'One turn of an agent, iterated as it runs.',
    guides: [GUIDES.running],
    items: ['AgentRun'],
    intro: {
      python: `\`Agent.run()\` and \`Agent.run_stream()\` return an \`AgentRun\`, the turn before it has started. \`LangModel.infer_stream()\` returns one too. Iterate it with \`async for\`.

Nothing runs until the first item is asked for, and the turn only goes as far as it has been iterated. Breaking out of the loop, or \`aclose()\`, ends the turn where it stands.

What it yields depends on where it came from. \`run()\` yields a [\`MessageOutput\`](../outputs/#messageoutput) per complete message: the model's, then each tool's result. \`run_stream()\` and \`infer_stream()\` yield [\`MessageDeltaOutput\`](../outputs/#messagedeltaoutput)s: pieces of the model's message as it's generated, and each tool's result whole. Errors from the model, a tool's provider or the console are raised from the loop.`,
      nodejs: `\`agent.run()\` and \`agent.runStream()\` return an \`AgentRun\`, the turn before it has started, typed as an \`AsyncIterableIterator\`. Iterate it with \`for await\`.

Nothing runs until the first \`next()\`, and the turn only goes as far as it has been iterated. Breaking out of the loop, or \`return()\`, ends the turn where it stands.

What it yields depends on where it came from. \`run()\` yields a [\`MessageOutput\`](../outputs/#messageoutput) per complete message: the model's, then each tool's result. \`runStream()\` yields [\`MessageDeltaOutput\`](../outputs/#messagedeltaoutput)s: pieces of the model's message as it's generated, and each tool's result whole. Errors from the model, a tool's provider or the console reject the \`next()\` they happen in, which \`for await\` throws.`,
    },
  },
  {
    slug: 'lang-model',
    title: 'LangModel',
    summary: 'A model run directly on a conversation, without an agent.',
    guides: [GUIDES.models],
    items: ['LangModel'],
    intro: `A \`LangModel\` is a model on its own: no history, no turn loop, and no tools run. \`create(desc)\` builds it through the provider set for the desc's org, the way an agent builds its model, and \`infer()\` runs it once on the messages you pass.

When the model asks for a tool, the request comes back in the message's \`tool_calls\`. Running the tool, and sending its result back in the next call, is up to you. Use a \`LangModel\` for one-off calls, or as what a factory given to [{{set_lang_model_provider}}](../model-providers/) returns.`,
  },
  {
    slug: 'model-providers',
    title: 'Model providers',
    summary: 'How a model desc is built: the backend set for each org, or a model of your own.',
    guides: [GUIDES.models],
    items: ['set_lang_model_provider', 'get_lang_model_provider', 'remove_lang_model_provider', 'Backend', 'CustomLangModel', 'StreamingCustomLangModel', 'LangModelFactory'],
    intro: `An agent or a \`LangModel\` names its model by a desc, \`<org>/<name>\`, or \`<org>\` alone for a CLI's default model. The provider set for the org decides how the model runs: the org's CLI, its own API, Bedrock, OpenRouter, vLLM, or a factory of your own. The same desc runs through every backend, except that DeepSeek's own API names its models without a version, so it takes DeepSeek's own names (\`deepseek-ai/deepseek-chat\`).

| Models | Naming | Desc |
| --- | --- | --- |
| Claude | \`anthropic/<official API id>\` | \`anthropic/claude-sonnet-5-5\` |
| GPT | \`openai/<official API id>\` | \`openai/gpt-6.1-sol\` |
| Gemini | \`google/<official API id>\` | \`google/gemini-3.1-pro\` |
| DeepSeek | Hugging Face repo id; DeepSeek's own name on its own API | \`deepseek-ai/DeepSeek-V3.2\`, \`deepseek-ai/deepseek-chat\` |
| Kimi | Hugging Face repo id | \`moonshotai/Kimi-K2-Thinking\` |
| GLM | Hugging Face repo id | \`zai-org/GLM-4.6\` |
| Qwen | Hugging Face repo id | \`Qwen/Qwen3-235B-A22B-Instruct-2507\` |

Each org starts with a provider. The default is the CLI where there is one (Claude Code for \`anthropic\`, Codex for \`openai\`, Gemini CLI for \`google\`), otherwise the org's own API.

| Org | CLI | APIs |
| --- | --- | --- |
| \`anthropic\` | Claude Code | own API, Bedrock, OpenRouter |
| \`openai\` | Codex | own API, Bedrock, OpenRouter |
| \`google\` | Gemini CLI | own API, OpenRouter |
| \`deepseek-ai\` | none | own API, Bedrock, OpenRouter, vLLM |
| \`moonshotai\` | none | own API, Bedrock, OpenRouter, vLLM |
| \`zai-org\` | none | own API, Bedrock, OpenRouter, vLLM |
| \`Qwen\` | none | own API, Bedrock, OpenRouter, vLLM |

The own APIs read their key from \`ANTHROPIC_API_KEY\`, \`OPENAI_API_KEY\`, \`GEMINI_API_KEY\`, \`DEEPSEEK_API_KEY\`, \`KIMI_API_KEY\`, \`ZAI_API_KEY\` and \`DASHSCOPE_API_KEY\`. Bedrock reads \`AWS_BEARER_TOKEN_BEDROCK\` (and \`AWS_REGION\`), OpenRouter reads \`OPENROUTER_API_KEY\`, and vLLM uses the server at \`VLLM_BASE_URL\` (by default \`http://localhost:8000/v1\`), with \`VLLM_API_KEY\` if it's set. Every backend but the CLIs needs a name in the desc.

Providers are process-wide: setting one changes how every agent and \`LangModel\` built afterwards runs that org's models.`,
  },
  {
    slug: 'tools',
    title: 'Tools and MCP',
    summary: 'Registering your own functions and MCP servers as tools an agent can call.',
    guides: [GUIDES.tools, GUIDES.mcp],
    items: ['register_tool', 'set_tool_provider', 'remove_tool_provider', 'register_mcp_stdio', 'register_mcp_streamable_http', 'unregister_mcp', 'ToolDesc', 'WebSearchEngine'],
    intro: {
      python: `A tool has two halves. Its [\`ToolDesc\`](#tooldesc), the name, description and parameter schema, is what the model sees, and it goes on the agent with \`AgentBuilder.tool()\` or \`tools()\`. Its provider, what runs when the model calls it, is set process-wide under the tool's name. When an agent is built, each desc is built through the provider set for its name, and a name with none fails \`build()\`.

The registry starts with every built-in tool (\`shell\`, \`read\`, \`write\`, \`edit\`, \`apply_patch\`, \`web_search\`, \`web_fetch\` and the rest), which the builder's \`system_tools()\`, \`shell_tool()\`, \`web_search_tool()\` and \`web_fetch_tool()\` add. \`register_tool()\` adds a function of your own, and \`register_mcp_stdio()\` and \`register_mcp_streamable_http()\` add an MCP server's tools. Each of these hands back the descs to give the agent.

A Python tool function is called with the model's arguments as keyword arguments when they're a dict, or else as one positional argument. It can be async. It returns what a message can hold: \`None\`, a bool, a number, a string, or a list or dict of those. A raise doesn't end the turn: it becomes the tool's result, \`"error: ValueError: ..."\`, so the model can try differently. A tool function isn't handed the agent's console; one that runs commands holds its own \`ConsoleClient\`.`,
      nodejs: `A tool has two halves. Its [\`ToolDesc\`](#tooldesc), the name, description and parameter schema, is what the model sees, and it goes on the agent with \`AgentBuilder.tool()\` or \`tools()\`. Its provider, what runs when the model calls it, is set process-wide under the tool's name. When an agent is built, each desc is built through the provider set for its name, and a name with none rejects \`build()\`.

The registry starts with every built-in tool (\`shell\`, \`read\`, \`write\`, \`edit\`, \`apply_patch\`, \`web_search\`, \`web_fetch\` and the rest), which the builder's \`systemTools()\`, \`shellTool()\`, \`webSearchTool()\` and \`webFetchTool()\` add. \`registerTool()\` adds a function of your own, and \`registerMcpStdio()\` and \`registerMcpStreamableHttp()\` add an MCP server's tools. Each of these hands back the descs to give the agent.

A JavaScript tool function is called with the model's arguments, the object the desc's \`parameters\` schema describes. It can return a value or a promise of one. A throw or a rejection doesn't end the turn: it becomes the tool's result, \`"error: ..."\`, so the model can try differently.`,
    },
  },
  {
    slug: 'messages',
    title: 'Messages',
    summary: 'The shape of a message and its parts: text, tool calls, values and images.',
    guides: [GUIDES.messages],
    items: ['Message', 'Role', 'Part', 'TextPart', 'FunctionPart', 'PartFunction', 'ValuePart', 'ImagePart', 'EmbeddedImage', 'UrlImage', 'Value'],
    intro: {
      python: `A conversation is a list of [\`Message\`](#message)s. Each has a \`role\` and \`contents\`, a list of parts, each tagged by its \`type\`. A message is a plain dict in its JSON form. The \`TypedDict\`s in \`ailoy.types\` only spell out the keys for an editor or a type checker. Nothing checks a dict against them at runtime: the Rust side does, when the dict arrives.

A query can be a string, taken as the user's text, or a whole message. \`Agent.history\`, \`AgentBuilder.history()\` and \`LangModel.infer()\` all use this shape.`,
      nodejs: `A conversation is a list of [\`Message\`](#message)s. Each has a \`role\` and \`contents\`, a list of parts, each tagged by its \`type\`. A message is a plain object in its JSON form. These types only spell out the keys for an editor. Nothing checks an object against them at runtime: the Rust side does, when the object arrives.

A query can be a string, taken as the user's text, or a whole message. \`agent.history\`, \`AgentBuilder.history()\` and \`LangModel.infer()\` all use this shape.`,
    },
  },
  {
    slug: 'outputs',
    title: 'Outputs',
    summary: 'What a turn or a model call gives back: whole messages, streamed deltas, finish reasons and token usage.',
    guides: [GUIDES.running, GUIDES.messages],
    items: ['MessageOutput', 'MessageDeltaOutput', 'MessageDelta', 'PartDelta', 'FinishReason', 'TokenUsage'],
    intro: `\`run()\` and \`infer()\` give [\`MessageOutput\`](#messageoutput)s: a whole message, why it ended, and the tokens it took. {{run_stream}} and {{infer_stream}} give [\`MessageDeltaOutput\`](#messagedeltaoutput)s: pieces of the model's message as it's generated. The one with a \`finish_reason\` marks the end of a message. Tool results come whole, one output each, and report no usage.

To rebuild a message from its deltas, append each \`delta\`'s text and parts in order until a \`finish_reason\` arrives. The agent does this itself, so \`history\` holds whole messages either way.`,
  },
  {
    slug: 'specs-and-options',
    title: 'Specs and options',
    summary: 'An agent as data, and the options a model, a browser and the context manager take.',
    guides: [GUIDES.agents, GUIDES.context, GUIDES.browser],
    items: ['AgentSpec', 'LangModelOptions', 'ThinkingEffort', 'BrowserOptions', 'UserDataDir', 'ContextManagerOptions'],
    intro: {
      python: `[\`AgentSpec\`](#agentspec) is what makes an agent distinct (its model, instruction, tools, model options, search engines and skills) as one dict, for \`Agent.from_spec\`. [\`LangModelOptions\`](#langmodeloptions) are the settings every model call takes, as \`LangModel.infer()\` and a custom model receive them.

In Python, the browser and context manager settings are keyword arguments of \`AgentBuilder.browser()\` and \`AgentBuilder.context_manager()\`, documented there.`,
      nodejs: `[\`AgentSpec\`](#agentspec) is what makes an agent distinct (its model, instruction, tools, model options, search engines and skills) as one object, for \`Agent.fromSpec\`. [\`LangModelOptions\`](#langmodeloptions) are the settings every model call takes, as \`LangModel.infer()\` and a custom model receive them. \`BrowserOptions\` and \`ContextManagerOptions\` are the objects \`AgentBuilder.browser()\` and \`AgentBuilder.contextManager()\` take.`,
    },
  },
  {
    slug: 'errors',
    title: 'Errors',
    summary: 'How ailoy reports failures: what is raised, and where.',
    guides: [GUIDES.running],
    items: ['AiloyError'],
    intro: {
      python: `ailoy raises three kinds of error:

- [\`AiloyError\`](#ailoyerror) for everything ailoy itself reports: a model whose org has no provider, a tool with no provider, an API error, an agent used during a turn or after \`close()\`.
- The \`virtx\` package's own classes for console failures, so a refusal inside a turn is the same \`virtx.ConsoleRefused\`, with the same \`code\`, as one from \`ConsoleClient.exec\`. A console whose channel is gone raises \`virtx.ConsoleBroken\`.
- \`ValueError\` for an argument ailoy can't take: a builder used after \`build()\`, an unknown \`user_browser\` name, a dict that isn't a message.

A turn's errors are raised from the \`async for\` that iterates it. A tool function's own exception isn't raised: it becomes the tool's result (\`"error: ..."\`), and the turn goes on.`,
      nodejs: `ailoy has no error classes of its own. Every failure is an \`Error\` whose \`code\` says what kind it is:

| \`code\` | Meaning |
| --- | --- |
| \`AILOY_ERROR\` | Everything ailoy itself reports: a model whose org has no provider, a tool with no provider, an API error, an agent used during a turn or after \`close()\`. |
| \`INVALID_ARG\` | An argument ailoy can't take: a builder used after \`build()\`, an unknown \`userBrowser\`, a negative token count. |
| \`TIMED_OUT\`, \`NOT_FOUND\`, \`NOT_EXECUTABLE\`, \`IS_A_DIRECTORY\`, \`IO_FAILED\`, \`BOOT_FAILED\`, ... | A console refusal, by the name \`@brekkylab/virtx\` gives its number, the same as from \`ConsoleClient.exec\`. |
| \`CONSOLE_REFUSED\` | A console refusal whose number has no name, with the number in the message. |
| \`CONSOLE_BROKEN\` | A console whose channel is gone. |

\`\`\`ts
try {
  for await (const out of agent.run('hello')) console.log(out.message);
} catch (e) {
  if (e.code === 'AILOY_ERROR') console.error('ailoy:', e.message);
  else throw e;
}
\`\`\`

A turn's errors reject the iterator's \`next()\`, which \`for await\` throws. A tool function's own throw isn't thrown: it becomes the tool's result (\`"error: ..."\`), and the turn goes on.`,
    },
  },
];

const BROWSER = {
  isolated: 'Start a browser of the agent\'s own, apart from the person\'s: started on first use and closed with the agent. Off by default, which drives the person\'s own browser.',
  headless: 'With `isolated`: run it with no window, as on a server.',
  executable: 'With `isolated`: the Chromium-based browser to start, by its executable. By default one installed here (Chrome, Chromium, Edge or Brave).',
  profile: 'With `isolated`: the user data directory it keeps, so sign-ins last. A fresh one by default.',
  user_browser: {
    python: 'Without `isolated`: which of the person\'s browsers, by name, or any other Chromium-based browser by its user data directory (a path). With none, whichever turned remote debugging on last.',
    nodejs: 'Without `isolated`: which of the person\'s browsers, by name, or any other by `{ userDataDir }`. With none, whichever turned remote debugging on last.',
  },
  careful: 'Tell the agent to use sites as a person would, so sites checking for bots are less likely to turn it away.',
  start_url: 'The page the agent\'s tab opens on.',
};

const CONTEXT = {
  max_input_tokens: { doc: 'Truncate the history as a turn begins once the previous model call\'s input went past this many tokens. `0` turns truncation off. Truncation keeps the system message and the recent turns, and replaces each earlier tool result\'s contents with `"[context truncated]"`, keeping its `id`. Whole turns are never dropped.', default: '0' },
  preserve_recent_turns: { doc: 'How many recent user turns truncation leaves alone. The turn just begun is one of them, so `0` keeps it too.', default: '3' },
  max_images: { doc: 'How many images tool results may hold in the history at the start of a turn. Past it, the oldest become `"[image from an earlier turn removed]"`. Images a person attached are left alone. `0` turns the cap off.', default: '30' },
  preserve_recent_images: { doc: 'How many tool-result images are left once `max_images` is passed. They\'re removed several at a time, so the prompt cache breaks once in several turns rather than every turn.', default: '10' },
};

const MODEL_DESC = 'The model\'s desc: `<org>/<name>`, such as `"anthropic/claude-sonnet-5-5"`, or `<org>` alone for a CLI\'s default model. See [Model providers](../model-providers/).';
const MESSAGES = 'The conversation to run the model on, oldest first.';
const TOOLS = 'The tools the model may call, by their descs. The model only asks for them: nothing runs them.';
const OPTIONS = 'Thinking effort and output schema for this call.';
const TOOL_FUNC = {
  python: 'What runs when the model calls the tool: a function, sync or async, called with the model\'s arguments as keyword arguments.',
  nodejs: 'What runs when the model calls the tool: a function called with the model\'s arguments object, returning a value or a promise.',
};

export const FILL = {
  // AgentBuilder
  'AgentBuilder.__init__': { doc: 'A builder for the model `model` names, built through the provider set for its org when `build()` runs.', params: { model: MODEL_DESC } },
  'AgentBuilder.instruction': {
    doc: 'Set the system instruction: the system message the agent\'s history starts with, which shapes how the model works. A leading system message in `history()` replaces it.',
    params: { instruction: 'The system prompt.' },
  },
  'AgentBuilder.tool': {
    doc: 'Give the agent one tool, by its desc. The model sees the desc; `build()` builds the tool through the provider set for `desc.name`.',
    params: { desc: 'The tool\'s name, description and parameter schema, such as what {{register_tool}} hands back.' },
  },
  'AgentBuilder.tools': {
    doc: 'Give the agent several tools at once, such as the descs {{register_mcp_stdio}} hands back.',
    params: { descs: 'The tools\' descs, each built through the provider set for its name.' },
  },
  'AgentBuilder.system_tools': {
    doc: 'Give the agent the local-execution tools that suit its model\'s family: `shell`; for `openai`, `apply_patch`; for every other org, `read` (`read_file` for `google`), `write` and `edit`; and `imgread`, except for `deepseek-ai`, `moonshotai`, `zai-org` and `Qwen`, whose backends take tool results as text alone. These tools run in the console, so the agent needs {{console()}} too.',
  },
  'AgentBuilder.shell_tool': { doc: 'Give the agent only the `shell` tool, without the rest of `system_tools()`. It runs in the console, so the agent needs {{console()}} too.' },
  'AgentBuilder.web_search_tool': {
    override: true,
    doc: 'Give the agent the `web_search` tool, which searches the web through the engines named, or through every engine when none are.',
    params: { engines: 'The engines to search with. All of them when empty or left out.' },
  },
  'AgentBuilder.web_fetch_tool': { doc: 'Give the agent the `web_fetch` tool, which isn\'t part of `system_tools()`. It fetches one URL per call, pages long bodies through `offset`, and allows one request per second per host.' },
  'AgentBuilder.history': {
    doc: 'Start the agent with this conversation, for example to resume an earlier session. A leading system message here replaces the one `instruction()` would make; otherwise the instruction is put at the front of it.',
    params: { history: 'The earlier messages, oldest first, such as a previous agent\'s `history`.' },
  },
  'AgentBuilder.console': {
    more: 'The console needn\'t be started: the first batch of tool calls with a console tool starts it. Nothing makes a console for you. Without one, tools that don\'t need it still run, and console tools fail with an error.',
    params: { console: { python: 'A `virtx.ConsoleClient`.', nodejs: 'A `ConsoleClient` from `@brekkylab/virtx`.' } },
  },
  'AgentBuilder.browser': {
    more: { python: 'Nothing starts until the agent first uses the browser, or `Agent.start_browser()`.', nodejs: 'Nothing starts until the agent first uses the browser, or `agent.startBrowser()`.' },
    params: { ...BROWSER, options: 'How to reach the browser. Left out, the person\'s own browser, with `careful` on.' },
  },
  'AgentBuilder.memory': {
    more: 'Adds the `mem_search` and `mem_insert` tools. The store must already exist (`mem init`): a missing file is reported by the first memory tool call, not by `build()`. Memory tools run in the console, so the agent needs {{console()}} too.',
    params: { memfile: 'The memory store\'s path in the console.' },
  },
  'AgentBuilder.skill': {
    doc: 'Give the agent the skill in `dir`, a directory in the console holding a `SKILL.md`. At `build()`, its frontmatter `name` and `description` are read through the console and listed in the system message with the skill\'s location. The agent reads the rest itself when it needs it, so it needs a console and a file-reading tool. Skills are only added to the system message `instruction()` makes, never to one already in `history()`. Call it once per skill.',
    params: { dir: 'The skill\'s directory, in the console.' },
  },
  'AgentBuilder.context_manager': {
    doc: 'Set how the agent keeps its history within the model\'s context: truncating old tool results once the input grows past a limit, and capping the images tool results hold. Settings left out keep their defaults. Without this call the defaults apply anyway: no truncation, and the image cap.',
    params: { ...CONTEXT, options: 'The settings. Any left out keep their defaults.' },
  },
  'AgentBuilder.thinking_effort': {
    more: 'Left unset, the model or backend default applies.',
    params: { effort: '`"low"`, `"medium"` or `"high"`.' },
  },
  'AgentBuilder.output_schema': { params: { schema: 'A JSON schema object.' } },
  'AgentBuilder.build': {
    override: true,
    doc: "Build the agent: the model through the provider set for its org, each tool through the provider set for its name, and each skill's `SKILL.md` read through the console. Awaitable, since building the model and reading skills take time.",
    returns: 'The agent.',
    raises: { python: '`AiloyError` when the model\'s org or a tool has no provider, a skill is given without a console or its frontmatter can\'t be read, or the browser settings don\'t fit together. The console\'s own failure, such as `virtx.ConsoleRefused`, when a skill\'s `SKILL.md` can\'t be read. `ValueError` when the builder was already built.', nodejs: '`AILOY_ERROR` when the model\'s org or a tool has no provider, a skill is given without a console or its frontmatter can\'t be read, or the browser settings don\'t fit together. The console\'s code, such as `NOT_FOUND`, when a skill\'s `SKILL.md` can\'t be read. `INVALID_ARG` when the builder was already built.' },
  },

  // Agent
  'Agent.from_spec': {
    more: 'The spec gives the model, instruction, tools and skills. The rest of what a builder sets comes from the options here.',
    params: {
      spec: 'The agent as data. See [`AgentSpec`](../specs-and-options/#agentspec).',
      history: 'Earlier messages to start from, as `AgentBuilder.history()` takes them.',
      console: { python: 'A `virtx.ConsoleClient` for the console tools.', nodejs: 'A `ConsoleClient` from `@brekkylab/virtx` for the console tools.' },
      memory: 'The memory store\'s path in the console, as `AgentBuilder.memory()` takes it.',
      options: 'What the agent starts with besides its spec: its history, console and memory.',
    },
    returns: 'The agent.',
  },
  'Agent.history': { more: 'It holds whole messages, from streamed turns too: the system message, each query, and every message the model and the tools produced.' },
  'Agent.start_browser': {},
  'Agent.run': {
    params: { query: 'What to ask: a string, as the user\'s text, or a whole [`Message`](../messages/#message).' },
    returns: 'The turn, which runs as it\'s iterated. See [`AgentRun`](../agent-run/).',
  },
  'Agent.run_stream': {
    params: { query: 'What to ask: a string, as the user\'s text, or a whole [`Message`](../messages/#message).' },
    returns: 'The turn, which runs as it\'s iterated. See [`AgentRun`](../agent-run/).',
  },
  'Agent.close': {},

  // AgentRun
  'AgentRun.aclose': { doc: 'End the turn where it stands, as breaking out of `async for` does.' },
  'AgentRun.next': { doc: 'Run the turn on to its next output. Done once the turn has ended.', returns: 'The next output, or done.' },
  'AgentRun.return': { returns: 'Done.' },

  // LangModel
  'LangModel.create': {
    doc: 'Build the model `desc` names, through the provider set for its org.',
    params: { desc: MODEL_DESC },
    returns: 'The model.',
    raises: { python: '`AiloyError` when the org has no provider, or the backend can\'t build the model.', nodejs: '`AILOY_ERROR` when the org has no provider, or the backend can\'t build the model.' },
  },
  'LangModel.infer': { params: { messages: MESSAGES, tools: TOOLS, options: OPTIONS }, returns: 'The model\'s whole message, why it ended, and the tokens it took.' },
  'LangModel.infer_stream': { params: { messages: MESSAGES, tools: TOOLS, options: OPTIONS }, returns: 'The model\'s message as deltas.' },

  // model providers
  set_lang_model_provider: {
    params: {
      org: 'The org, the part of a desc before the `/`.',
      provider: 'A backend\'s name (see [`Backend`](#backend)), or a factory called with the org and the name after it ({{None}} when the desc names only the org) that returns a `LangModel` or a [`CustomLangModel`](#customlangmodel).',
    },
  },
  get_lang_model_provider: { params: { desc: 'A desc, or just its org.' }, returns: { python: 'The backend\'s name, `"custom"` for a factory, or `None`.', nodejs: 'The backend\'s name, `\'custom\'` for a factory, or `null`.' } },
  remove_lang_model_provider: { more: 'Models of the org then fail to build until a provider is set again.', params: { org: 'The org.' }, returns: 'Whether the org had a provider.' },
  Backend: {
    doc: 'How a model is run.',
    more: '`"cli"` is the org\'s CLI (Claude Code, Codex or Gemini CLI); `"api"` the org\'s own API; `"bedrock"`, `"openrouter"` and `"vllm"` those services. Which backends an org has is in the tables above.',
  },
  CustomLangModel: { more: { python: 'Declared in the type stubs only, for type checking: import it under `TYPE_CHECKING` from `ailoy._ailoy`. Any object with these methods works.', nodejs: 'Its methods are called with `this` as the model.' } },
  'CustomLangModel.infer': { doc: 'Run the model on `messages` to completion. May return the output or an awaitable of it.', params: { messages: MESSAGES, tools: 'The tools the model may call.', options: 'The call\'s settings.' }, returns: 'The model\'s whole message.' },
  'CustomLangModel.infer_stream': { doc: 'Run the model on `messages` and yield its output as deltas while it\'s generated. Optional.', params: { messages: MESSAGES, tools: 'The tools the model may call.', options: 'The call\'s settings.' }, returns: 'The deltas.' },
  StreamingCustomLangModel: { doc: 'A `CustomLangModel` that streams: one with `infer_stream` as well as `infer`.', more: 'Declared in the type stubs only, for type checking.' },
  'StreamingCustomLangModel.infer_stream': { doc: 'Run the model on `messages` and yield its output as deltas while it\'s generated.', params: { messages: MESSAGES, tools: 'The tools the model may call.', options: 'The call\'s settings.' }, returns: 'The deltas.' },
  LangModelFactory: { doc: 'What `set_lang_model_provider` takes for a model of your own: called with the org and the name after it (`None` when the desc names only the org), it returns a `LangModel` or a `CustomLangModel`. Declared in the type stubs only, for type checking.' },

  // tools
  register_tool: { params: { desc: 'The tool\'s name, description and parameter schema. `desc.name` is what the model calls.', func: TOOL_FUNC }, returns: 'The same `desc`, to give the agent.' },
  set_tool_provider: {
    doc: 'Build the tool `name` by calling `func`, replacing the provider it had. Unlike `register_tool`, it takes no desc: use it to change what runs behind a name the agent already has, such as a built-in tool.',
    params: { name: 'The tool\'s name.', func: TOOL_FUNC },
  },
  remove_tool_provider: { more: 'An agent built afterwards with the tool fails to build.', params: { name: 'The tool\'s name.' }, returns: 'Whether the tool had a provider.' },
  register_mcp_stdio: {
    params: { prefix: 'Put before each tool\'s name, as `{prefix}__{name}`. It\'s also how `unregister_mcp` names the server.', command: 'The server\'s executable.', args: { doc: 'The command\'s arguments.', default: 'none' } },
    returns: 'The descs of the server\'s tools, to give the agent with `tools()`.',
  },
  register_mcp_streamable_http: {
    params: { prefix: 'Put before each tool\'s name, as `{prefix}__{name}`. It\'s also how `unregister_mcp` names the server.', url: 'The server\'s streamable-HTTP endpoint.' },
    returns: 'The descs of the server\'s tools, to give the agent with `tools()`.',
  },
  unregister_mcp: { params: { prefix: 'The prefix the server was registered under.' }, returns: 'How many tools were removed.' },
  ToolDesc: { doc: 'A tool as the model sees it.' },
  'ToolDesc.name': 'The tool\'s name: what the model calls, and what its provider is set under.',
  'ToolDesc.description': 'What the tool does, in words, for the model.',
  'ToolDesc.parameters': 'A JSON schema of the arguments, usually an object schema: `{"type": "object", "properties": ...}`.',
  'ToolDesc.returns': 'A JSON schema of the result. Left out, the tool returns free-form text or JSON.',
  WebSearchEngine: { doc: 'An engine the `web_search` tool can use.' },

  // messages
  Message: { doc: 'One message of a conversation.' },
  'Message.role': 'Who wrote it.',
  'Message.contents': 'Its parts: text, values, images.',
  'Message.thinking': 'The model\'s thinking before it answered, for models that give it.',
  'Message.tool_calls': 'The tools the model asks to run, one [`FunctionPart`](#functionpart) each.',
  'Message.id': 'For a tool result, the id of the call it answers.',
  'Message.signature': 'A signature of `thinking`, for the APIs that need it sent back with the thinking.',
  Role: { doc: 'Who wrote a message: the system instruction, the user, the model (`assistant`) or a tool\'s result (`tool`).' },
  Part: { doc: 'One part of a message, tagged by `type`.' },
  TextPart: { doc: 'Text.' },
  'TextPart.type': '`"text"`.',
  'TextPart.text': 'The text.',
  FunctionPart: { doc: 'A tool call the model asks for.' },
  'FunctionPart.type': '`"function"`.',
  'FunctionPart.id': 'Pairs the call with its result, a tool message with the same `id`.',
  'FunctionPart.function': 'The tool and its arguments.',
  PartFunction: { doc: 'The tool a call names and its arguments.' },
  'PartFunction.name': 'The tool\'s name.',
  'PartFunction.arguments': 'The arguments, usually an object the tool\'s parameter schema describes.',
  ValuePart: { doc: 'A structured value, such as a tool\'s result.' },
  'ValuePart.type': '`"value"`.',
  'ValuePart.value': 'Any JSON value.',
  ImagePart: { doc: 'An image, embedded or by URL.' },
  'ImagePart.type': '`"image"`.',
  'ImagePart.image': 'The image.',
  EmbeddedImage: { doc: 'An image carried in the message.' },
  'EmbeddedImage.type': '`"embedded"`.',
  'EmbeddedImage.mime_type': 'Its type, such as `"image/png"`.',
  'EmbeddedImage.data': 'The image\'s bytes.',
  UrlImage: { doc: 'An image by its URL.' },
  'UrlImage.type': '`"url"`.',
  'UrlImage.url': 'Where the image is.',

  // outputs
  MessageOutput: { more: { python: 'What `Agent.run` yields and `LangModel.infer` returns.', nodejs: 'What `agent.run` yields and `LangModel.infer` returns.' } },
  'MessageOutput.message': 'The message.',
  'MessageOutput.finish_reason': 'Why it ended.',
  'MessageOutput.usage': 'The tokens the model call took. Only a model\'s messages have it; a tool result doesn\'t.',
  MessageDeltaOutput: {},
  'MessageDeltaOutput.delta': 'The piece of the message.',
  'MessageDeltaOutput.finish_reason': 'Set on the last piece of a message: why it ended.',
  'MessageDeltaOutput.usage': 'The tokens the model call took, when the backend reports them.',
  MessageDelta: { doc: 'A piece of a message, with the fields that have arrived in it.' },
  'MessageDelta.role': 'Who is writing, on the first piece.',
  'MessageDelta.contents': 'Pieces of the message\'s parts.',
  'MessageDelta.id': 'The message\'s id, for a tool result.',
  'MessageDelta.thinking': 'A piece of the model\'s thinking.',
  'MessageDelta.tool_calls': 'Pieces of the tool calls.',
  'MessageDelta.signature': 'The thinking\'s signature, for the APIs that send one.',
  'PartDelta.type': 'Which kind of part it\'s a piece of.',
  'PartDelta.text': 'A piece of the text, for a `text` part.',
  FinishReason: { more: '`stop` is a finished answer, `length` an answer the output limit cut short, `tool_call` an answer that calls tools, and `refusal` one the model refused, with its `reason`.' },
  'FinishReason.type': 'Why the message ended.',
  'FinishReason.reason': 'For `refusal`, why.',
  TokenUsage: { doc: 'The tokens a model call took.' },
  'TokenUsage.input_tokens': 'Every token of input the call processed, cached or not.',
  'TokenUsage.output_tokens': 'The tokens the model wrote.',
  'TokenUsage.cache_creation_input_tokens': 'The part of `input_tokens` written to a prompt cache.',
  'TokenUsage.cache_read_input_tokens': 'The part of `input_tokens` read from a prompt cache.',

  // specs and options
  AgentSpec: { doc: 'What makes an agent distinct, as data: its model, instruction and tools. The console, browser, memory and history aren\'t part of it.' },
  'AgentSpec.model': MODEL_DESC,
  'AgentSpec.instruction': 'The system instruction.',
  'AgentSpec.tools': 'The tools\' descs. Each name needs a provider set for it.',
  'AgentSpec.model_options': 'Thinking effort and output schema.',
  'AgentSpec.web_search_engines': 'The engines `web_search` uses, when it\'s among `tools`. All of them when left out.',
  'AgentSpec.skills': 'Skill directories in the console, each holding a `SKILL.md`.',
  LangModelOptions: { doc: 'The settings every model call takes, whichever backend runs it.' },
  'LangModelOptions.thinking_effort': 'How much the model thinks. Left out, the model or backend default.',
  'LangModelOptions.output_schema': 'A JSON schema the output must match.',
  ThinkingEffort: { doc: 'How much the model thinks before it answers.' },
  BrowserOptions: { doc: 'How `AgentBuilder.browser()` reaches a browser: the person\'s own, with remote debugging on, or with `isolated` one of the agent\'s own.' },
  ...Object.fromEntries(Object.entries(BROWSER).map(([k, v]) => [`BrowserOptions.${k}`, v])),
  UserDataDir: {},
  'UserDataDir.user_data_dir': 'The browser\'s user data directory.',
  ContextManagerOptions: { doc: 'The settings `AgentBuilder.contextManager()` takes. Any left out keep their defaults.' },
  ...Object.fromEntries(Object.entries(CONTEXT).map(([k, v]) => [`ContextManagerOptions.${k}`, v])),

  // errors
  AiloyError: { doc: 'The exception for everything ailoy itself reports: a model or tool with no provider, an API error, an agent used during a turn or after `close()`. Its message says what failed.' },
};

export { RAISES };
