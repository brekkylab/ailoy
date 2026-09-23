// What the main panel can be showing. `settings` has no sidebar row: it is reached from
// the gear in the title bar.
export type MainView = "session" | "context" | "agent" | "artifacts" | "settings";

// What a selected context can be opened on, listed under the contexts in the sidebar.
export type ContextCommand = "files";

// The pages of the agent editor, listed under the agents in the sidebar.
export type AgentSection = "general" | "prompt" | "tools" | "sandbox" | "advanced";
