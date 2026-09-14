// The conversation: everything the engine has written down, then whatever the current
// run has produced since.
//
// The two halves overlap in exactly one place, and the overlap is deliberate. The engine
// persists the assistant message — and emits its `message` event — when the *model turn*
// ends, which is before any of that turn's tools have run; each tool answer is then a
// separate `role: "tool"` row written as it finishes. So mid-run the stored assistant
// message already carries `tool_calls` whose results do not exist yet, while the live run
// holds those same calls with their real status.
//
// `claimed` is the seam. Every call id named by a stored assistant message belongs to its
// bubble, which reads the live entry for that id when there is one; the live block below
// renders only the ids storage has not claimed yet. Exactly one card per call, at every
// moment of a run.
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
  const ackMessages = useRunStore((s) => s.ackMessages);
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

  // A persisted message invalidates the list. What is acked afterwards is the version
  // captured *before* the refetch, not whatever the counter reads when it resolves: a
  // message that arrives while the request is in flight bumps the counter past `v`, so
  // acking `v` still leaves version > acked and this runs again for it. Acking the live
  // value instead would mark that message fetched when it was not, and the only thing
  // that ever brought it in would be some later, unrelated event.
  const { messagesVersion, messagesAcked } = live;
  useEffect(() => {
    if (!sessionId || messagesVersion === messagesAcked) return;
    const v = messagesVersion;
    void qc.invalidateQueries({ queryKey: ["messages", sessionId] }).then(() => ackMessages(sessionId, v));
  }, [sessionId, messagesVersion, messagesAcked, qc, ackMessages]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.data?.length, live.text, live.thinking, live.toolOrder.length]);

  const { turns, toolResults, claimed, lastAssistantSeq } = useMemo(() => {
    const toolResults = new Map<string, StoredMessage>();
    const turns: StoredMessage[] = [];
    const claimed = new Set<string>();
    let lastAssistantSeq: number | null = null;
    for (const m of messages.data ?? []) {
      if (m.depth !== 0) continue; // sub-agent internals stay hidden in v1
      // A tool row is an answer to a call, never a turn of its own. One without an `id`
      // cannot be matched to its call, but it is still not a bubble.
      if (m.message.role === "tool") {
        if (m.message.id) toolResults.set(m.message.id, m);
        continue;
      }
      if (m.message.role === "system") continue;
      if (m.message.role === "assistant") {
        lastAssistantSeq = m.seq;
        for (const p of m.message.tool_calls ?? []) if (p.type === "function") claimed.add(p.id);
      }
      turns.push(m);
    }
    return { turns, toolResults, claimed, lastAssistantSeq };
  }, [messages.data]);

  // A session id can outlive the session: the one in `localStorage` after the row was
  // deleted, or one the list has not dropped yet. The engine says `not_found`; there is
  // nothing to compose into, so this reads as "no session" rather than as a failure.
  const gone = messages.isError && api.kindOf(messages.error) === "not_found";
  if (!sessionId || gone) return <EmptyState text={S.noSession} />;

  const streaming = live.status === "running";
  // Calls the engine announced before it wrote the message that names them. Once that
  // message lands they move up into its bubble and this list empties.
  const unclaimed = streaming ? live.toolOrder.filter((id) => !claimed.has(id)) : [];
  // A run that has produced nothing yet still says so.
  const pulse = streaming && !live.text && !live.thinking && live.toolOrder.length === 0;
  const showLive = streaming && (pulse || !!live.text || !!live.thinking || unclaimed.length > 0);
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
              <AssistantBubble
                key={m.seq}
                message={m.message}
                toolResults={toolResults}
                live={live}
                isLatest={m.seq === lastAssistantSeq}
              />
            ),
          )}
          {showLive && (
            <div className="max-w-[92%] space-y-1">
              {live.thinking && (
                <pre className="max-h-32 overflow-auto rounded bg-muted/40 p-2 text-xs whitespace-pre-wrap text-muted-foreground">
                  {live.thinking}
                </pre>
              )}
              {live.text && <Markdown text={live.text} />}
              {unclaimed.map((id) => {
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
              {pulse && <span className="animate-pulse text-sm text-muted-foreground">…</span>}
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
