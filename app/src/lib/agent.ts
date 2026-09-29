// The editor's model for an agent, and its serialization into the two shapes that own it:
//
//   AgentSpec  src/agent/spec.rs              model, instruction, tools, subagents
//   Recipe     cortex/src/rootfs/recipe.rs    base image and steps — the editor's "sandbox"
//
// They stay separate because ailoy keeps runtime (sandbox, MCP servers, mounted context)
// off the spec.

// ── The sandbox ──────────────────────────────────────────────────

/** cortex's four build instructions. */
export type StepKind = "run" | "copy" | "env" | "workdir";

/**
 * A step as the editor holds it: flat, so switching a row's kind keeps what was typed in
 * the other field. `recipe()` converts it to cortex's shape.
 */
export interface StepRow {
  id: string;
  kind: StepKind;
  /** `run`'s command, `copy`'s src, `env`'s key, `workdir`'s directory. */
  first: string;
  /** `copy`'s dst and `env`'s value; unused by `run` and `workdir`. */
  second: string;
}

/** cortex's `Step`, tagged by name: `{"run": …}`, `{"copy": {…}}`. */
export type Step =
  | { run: string }
  | { copy: { src: string; dst: string } }
  | { env: { key: string; value: string } }
  | { workdir: string };

/** The format version cortex reads; it refuses any other. */
const RECIPE_FORMAT = 1;

export const STEP_KINDS: { id: StepKind; label: string; first: string; second: string | null }[] = [
  { id: "run", label: "RUN", first: "Command", second: null },
  { id: "copy", label: "COPY", first: "Source", second: "Destination" },
  { id: "env", label: "ENV", first: "Key", second: "Value" },
  { id: "workdir", label: "WORKDIR", first: "Directory", second: null },
];

/** One step as cortex's `Display` writes it, i.e. the line the build reports. */
export function stepLine(step: StepRow): string {
  switch (step.kind) {
    case "run":
      return `RUN ${step.first}`;
    case "copy":
      return `COPY ${step.first} ${step.second}`;
    case "env":
      return `ENV ${step.first}=${step.second}`;
    case "workdir":
      return `WORKDIR ${step.first}`;
  }
}

// ── Tools ────────────────────────────────────────────────────────

/**
 * The builtins ailoy ships (`src/tool/impl/builtins`), by registered name. Every agent
 * gets all of them, so they are written to the spec and not shown in the editor.
 */
export const BUILTIN_TOOLS = ["read", "write", "edit", "apply_patch", "shell", "web_search", "web_fetch"];

/**
 * `WebSearchEngineKind::ALL`. An empty selection means all, i.e. `web_search_engines: None`
 * in the spec.
 */
export const SEARCH_ENGINES = [
  "Bing",
  "Brave",
  "DuckDuckGo",
  "Google",
  "Mojeek",
  "Naver",
  "Startpage",
  "Yahoo",
  "Yandex",
];

/** `MCPToolProviderElem`. ailoy registers the stdio variant but does not implement it. */
export interface McpServer {
  id: string;
  name: string;
  transport: "http" | "stdio";
  /** A URL for `http`, a command line for `stdio`. */
  target: string;
}

// ── An agent ─────────────────────────────────────────────────────

/**
 /**
  * Strings, not numbers: each is `Option<…>` in ailoy, and an empty field must stay empty
  * rather than coerce to `0`.
  */
export interface Options {
  temperature: string;
  topP: string;
  topK: string;
  maxTokens: string;
}

export interface Agent {
  /** Also the name of its directory under the cache. */
  id: string;
  /** Also the agent's `card.name` when it is used as a sub-agent. */
  name: string;
  description: string;
  /** `provider/model`, resolved against ailoy's provider registry. */
  model: string;
  /** The system prompt. `AgentSpec::instruction`. */
  instruction: string;
  /** Empty means every engine. */
  engines: string[];
  mcp: McpServer[];
  /** Ids of other agents, whose specs are inlined on serialize. */
  subagents: string[];
  /** The id of the context mounted read-only into this agent's sandbox, if any. At most one. */
  context: string | null;
  options: Options;
  /** Handed to the console as a cortex `Recipe`. */
  sandbox: { base: string; steps: StepRow[] };
}

/**
 * Distinct per page load, so ids minted now cannot collide with ones written before a
 * reload (a counter alone restarts at zero).
 */
const RUN = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
let counter = 0;
/** An id made only of characters the backend accepts in a directory name: letters, digits, `-`, `_`. */
export function newId(prefix: string): string {
  counter += 1;
  return `${prefix}-${RUN}-${counter.toString(36)}`;
}

export function blankAgent(name: string, model: string): Agent {
  return {
    id: newId("agent"),
    name,
    description: "",
    model,
    instruction: "",
    engines: [],
    mcp: [],
    subagents: [],
    context: null,
    options: { temperature: "", topP: "", topK: "", maxTokens: "" },
    sandbox: {
      base: "python:3.13-slim",
      steps: [{ id: newId("step"), kind: "workdir", first: "/workspace", second: "" }],
    },
  };
}

/** A copy under a new id, with every row re-keyed so the two never share one. */
export function duplicateOf(agent: Agent, name: string): Agent {
  return {
    ...agent,
    id: newId("agent"),
    name,
    mcp: agent.mcp.map((server) => ({ ...server, id: newId("mcp") })),
    sandbox: { base: agent.sandbox.base, steps: agent.sandbox.steps.map((step) => ({ ...step, id: newId("step") })) },
  };
}

/**
 * One stored document as the editor's model. Fields are laid over a blank agent, so a
 * document from an older version or edited by hand still opens. The collection's envelope
 * (`default`, `createdAt`, `updatedAt`) is dropped.
 */
export function fromStored(doc: Record<string, unknown>): Agent {
  // Legacy `contexts` list: its first entry becomes `context`.
  const { default: _d, createdAt: _c, updatedAt: _u, contexts, ...rest } = doc;
  const stored = rest as Partial<Agent> & { id: string };
  const blank = blankAgent(stored.name ?? "", "");
  return {
    ...blank,
    ...stored,
    id: stored.id,
    context: stored.context ?? (Array.isArray(contexts) ? ((contexts[0] as string | undefined) ?? null) : null),
    // Nested fields merged explicitly: a shallow spread would leave `sandbox.steps` missing
    // on a document without a sandbox, and a step without `id` unkeyable.
    options: { ...blank.options, ...(stored.options ?? {}) },
    sandbox: {
      base: stored.sandbox?.base ?? blank.sandbox.base,
      steps: (stored.sandbox?.steps ?? blank.sandbox.steps).map((step, i) => ({
        ...step,
        id: step.id || `step-${stored.id}-${i}`,
      })),
    },
  };
}

// ── Serializing ──────────────────────────────────────────────────

function num(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/** Whether a sampling field holds something `spec()` would drop. */
export const badNumber = (value: string) => value.trim() !== "" && num(value) === undefined;

/**
 /**
  * The steps as cortex would store them. Rows with an empty first field are still being
  * typed and are dropped.
  */
export function recipe(agent: Agent): { v: number; base: string; steps: Step[] } {
  const steps: Step[] = [];
  for (const step of agent.sandbox.steps) {
    const first = step.first.trim();
    const second = step.second.trim();
    if (!first) continue;
    if (step.kind === "run") steps.push({ run: first });
    else if (step.kind === "workdir") steps.push({ workdir: first });
    else if (step.kind === "copy" && second) steps.push({ copy: { src: first, dst: second } });
    else if (step.kind === "env") steps.push({ env: { key: first, value: second } });
  }
  return { v: RECIPE_FORMAT, base: agent.sandbox.base.trim(), steps };
}

/**
 /**
  * One `AgentSpec`, with sub-agents inlined. Tools are bare names; the provider resolves
  * each to the factory owning its description and schema. `depth` stops cycles, since two
  * agents may name each other.
  */
export function spec(agent: Agent, all: Agent[], depth = 0): Record<string, unknown> {
  const out: Record<string, unknown> = { model: agent.model.trim() };
  if (agent.instruction.trim()) out.instruction = agent.instruction;
  out.tools = BUILTIN_TOOLS.map((name) => ({ name }));

  if (depth < 4 && agent.subagents.length) {
    const subs = agent.subagents
      .map((id) => all.find((other) => other.id === id))
      .filter((sub): sub is Agent => !!sub && sub.id !== agent.id)
      .map((sub) => spec(sub, all, depth + 1));
    if (subs.length) out.subagents = subs;
  }

  const options: Record<string, number> = {};
  const set = (key: string, value: string) => {
    const n = num(value);
    if (n !== undefined) options[key] = n;
  };
  set("max_tokens", agent.options.maxTokens);
  set("temperature", agent.options.temperature);
  set("top_p", agent.options.topP);
  set("top_k", agent.options.topK);
  if (Object.keys(options).length) out.model_options = options;

  // The card is what a calling agent reads; written whenever there is something to say,
  // since any agent may be picked as a sub-agent elsewhere.
  if (agent.name.trim() || agent.description.trim()) {
    out.card = { name: agent.name.trim(), description: agent.description.trim(), skills: [] };
  }

  // A full selection is written as none: same meaning, and it does not go stale when ailoy
  // adds an engine.
  if (agent.engines.length && agent.engines.length < SEARCH_ENGINES.length) {
    out.web_search_engines = [...agent.engines];
  }
  return out;
}

/** What one agent runs from: the spec plus its runtime. */
export function entry(agent: Agent, all: Agent[]): Record<string, unknown> {
  const out: Record<string, unknown> = { spec: spec(agent, all), sandbox: recipe(agent) };
  if (agent.context) out.context = agent.context;
  const servers = agent.mcp.filter((server) => server.name.trim() && server.target.trim());
  if (servers.length) {
    out.mcp = servers.map((server) =>
      server.transport === "http"
        ? { name: server.name.trim(), streamable_http: { url: server.target.trim() } }
        : { name: server.name.trim(), stdio: { command: server.target.trim() } },
    );
  }
  return out;
}
