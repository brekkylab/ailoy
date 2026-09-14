// The conversation: everything the engine has written down, then whatever the current
// run has produced since.
//
// Those two halves never overlap. The engine emits a `message` event the moment it
// persists something, and the store's reducer clears the matching live text on it, so the
// live block is only ever the tail that storage has not caught up with. The live block is
// gated on `status === "running"` for the same reason from the other side: after a run
// ends, its tool cards live in the refetched `AssistantBubble`, and keeping the live ones
// up would show every call twice until the query settled.
//
// The engine owns the message list; this component owns exactly one query key,
// `["messages", id]`. `["sessions"]` and `["usage", id]` belong to `Composer`/`UsageBar`,
// which watch the run's start and its terminal transition.

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";

import * as api from "@/api";
import { Composer } from "@/components/Composer";
import { Markdown } from "@/components/Markdown";
import { AssistantBubble, UserBubble } from "@/components/MessageBubble";
import { ToolCallCard } from "@/components/ToolCallCard";
import { attachRun } from "@/events";
import { selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";
import type { StoredMessage } from "@/types";

function EmptyState({ text }: { text: string }) {
  return <div className="grid min-h-0 flex-1 place-items-center text-sm text-muted-foreground">{text}</div>;
}

export function Thread({ sessionId }: { sessionId: string | null }) {
  const qc = useQueryClient();
  const live = useRunStore(selectRun(sessionId));
  const clearDirty = useRunStore((s) => s.clearDirty);
  const messages = useQuery({
    queryKey: ["messages", sessionId],
    queryFn: () => api.messageList(sessionId!),
    enabled: sessionId != null,
  });
  const bottom = useRef<HTMLDivElement>(null);

  // After a reload, or when the window comes back to a session, a run may still be going.
  // `attachRun` is a no-op when this session already has a live channel.
  useEffect(() => {
    if (sessionId) void attachRun(sessionId).catch(() => {});
  }, [sessionId]);

  // A persisted message invalidates the list. The flag is cleared only once the refetch
  // has resolved: clearing it first would open a window in which the store says "clean"
  // while the query still holds the message that is missing, and a second event arriving
  // in that window would be the only thing that ever brought it in.
  const dirty = live.messagesDirty;
  useEffect(() => {
    if (!sessionId || !dirty) return;
    void qc.invalidateQueries({ queryKey: ["messages", sessionId] }).then(() => clearDirty(sessionId));
  }, [sessionId, dirty, qc, clearDirty]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.data?.length, live.text, live.thinking, live.toolOrder.length]);

  const { turns, toolResults } = useMemo(() => {
    const toolResults = new Map<string, StoredMessage>();
    const turns: StoredMessage[] = [];
    for (const m of messages.data ?? []) {
      if (m.depth !== 0) continue; // sub-agent internals stay hidden in v1
      if (m.message.role === "tool" && m.message.id) {
        toolResults.set(m.message.id, m);
        continue;
      }
      if (m.message.role === "system") continue;
      turns.push(m);
    }
    return { turns, toolResults };
  }, [messages.data]);

  // A session id can outlive the session: the one in `localStorage` after the row was
  // deleted, or one the list has not dropped yet. The engine says `not_found`; there is
  // nothing to compose into, so this reads as "no session" rather than as a failure.
  const gone = messages.isError && api.kindOf(messages.error) === "not_found";
  if (!sessionId || gone) return <EmptyState text={S.noSession} />;

  const streaming = live.status === "running";
  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.isError && (
            <div className="text-sm text-destructive">
              {S.loadFailed} ({api.messageOf(messages.error)})
            </div>
          )}
          {turns.map((m) =>
            m.message.role === "user" ? (
              <UserBubble key={m.seq} message={m.message} />
            ) : (
              <AssistantBubble key={m.seq} message={m.message} toolResults={toolResults} />
            ),
          )}
          {streaming && (
            <div className="max-w-[92%] space-y-1">
              {live.thinking && (
                <pre className="max-h-32 overflow-auto rounded bg-muted/40 p-2 text-xs whitespace-pre-wrap text-muted-foreground">
                  {live.thinking}
                </pre>
              )}
              {live.text && <Markdown text={live.text} />}
              {live.toolOrder.map((id) => {
                const c = live.toolCalls[id];
                return (
                  <ToolCallCard
                    key={id}
                    name={c.name}
                    args={c.arguments}
                    status={c.status}
                    result={c.result}
                    startedAt={c.startedAt}
                  />
                );
              })}
              {!live.text && !live.thinking && live.toolOrder.length === 0 && (
                <span className="animate-pulse text-sm text-muted-foreground">…</span>
              )}
            </div>
          )}
          {live.status === "error" && live.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">
              {S.errorPrefix} ({live.error.kind}): {live.error.message}
            </div>
          )}
          <div ref={bottom} />
        </div>
      </div>
      <Composer sessionId={sessionId} />
    </>
  );
}
