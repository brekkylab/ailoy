export interface SessionSummary {
  id: string;
  title: string;
  /** The agent it runs with; `null` means the default agent. */
  agent: string | null;
  updated_at: number;
  running?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  created_at: number;
}

export interface Provider {
  key: string;
  label: string;
}

export interface Model {
  id: string;
  name: string;
  /** A `Provider.key`. */
  provider: string;
}
