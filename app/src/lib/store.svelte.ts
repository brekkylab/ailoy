// The app's state, in memory. Stands in for what the backend will own later: sessions,
// their messages, and the key/model settings.

import { DEFAULT_BEDROCK_REGION, MODELS, seedMessages, seedSessions } from "@/lib/mock";
import { S } from "@/strings";
import type { ChatMessage, SessionSummary } from "@/types";

let counter = 0;
const uid = (p: string) => `${p}${Date.now().toString(36)}${(counter++).toString(36)}`;

class Store {
  sessions = $state<SessionSummary[]>(seedSessions());
  messages = $state<Record<string, ChatMessage[]>>(seedMessages());
  running = $state<Record<string, boolean>>({});
  /** Provider key → API key. For Bedrock, the bearer token. */
  keys = $state<Record<string, string>>({});
  bedrockRegion = $state(DEFAULT_BEDROCK_REGION);
  defaultModel = $state(MODELS[0].id);

  private touch(id: string) {
    const s = this.sessions.find((x) => x.id === id);
    if (!s) return;
    s.updated_at = Date.now();
    this.sessions = [s, ...this.sessions.filter((x) => x.id !== id)];
  }

  /** Sends into `sessionId`, or creates a session from a draft when it is `null`. */
  send(sessionId: string | null, text: string, agent: string | null): string {
    let id = sessionId;
    if (!id) {
      id = uid("s");
      const title = text.length > 40 ? `${text.slice(0, 40)}…` : text;
      this.sessions = [{ id, title, agent, updated_at: Date.now() }, ...this.sessions];
    }
    const sid = id;
    this.messages[sid] = [...(this.messages[sid] ?? []), { id: uid("m"), role: "user", text, created_at: Date.now() }];
    this.touch(sid);
    this.running[sid] = true;
    setTimeout(() => {
      if (!this.running[sid]) return;
      this.messages[sid] = [
        ...(this.messages[sid] ?? []),
        { id: uid("m"), role: "assistant", text: S.notConnected, created_at: Date.now() },
      ];
      this.running[sid] = false;
    }, 700);
    return sid;
  }

  stop(id: string) {
    this.running[id] = false;
  }

  rename(id: string, title: string) {
    const s = this.sessions.find((x) => x.id === id);
    if (s) s.title = title;
  }

  remove(id: string) {
    this.sessions = this.sessions.filter((x) => x.id !== id);
    delete this.messages[id];
  }

  setAgent(id: string, agent: string) {
    const s = this.sessions.find((x) => x.id === id);
    if (s) s.agent = agent;
  }
}

export const store = new Store();
