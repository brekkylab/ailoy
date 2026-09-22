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
// `claimed` is the seam. Every call id named by a stored assistant message belongs to the
// group that message's calls fall into, which reads the live entry for that id when there
// is one; the ids storage has not claimed yet are folded onto the end of that group, or
// drawn by the live block below when there is streamed text above them. Exactly one card
// per call, at every moment of a run.
//
// Which calls share a group is `lib/thread`'s question — adjacent calls collapse into one
// row, and anything the model *says* closes it — and it answers it from the stored list
// alone, which is why that half is tested without a webview.
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
import { ToolGroup } from "@/components/ToolGroup";
import { attachRun } from "@/events";
import { buildThread, liveGroupKey, resolveCall, withLiveCalls, type GroupCall } from "@/lib/thread";
import { selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";

/**
 * How far from the end still counts as "at the end", in px. Wide enough to survive the
 * rounding a zoomed webview puts on `scrollHeight`, and the half-line a fresh chunk of
 * streamed text adds between the scroll event and the effect that reads this.
 */
const BOTTOM_SLACK = 80;

function EmptyState({ text }: { text: string }) {
  return <div className="grid min-h-0 flex-1 place-items-center text-sm text-muted-foreground">{text}</div>;
}

export function Thread({
  sessionId,
  draft,
  onCreated,
}: {
  sessionId: string | null;
  /**
   * A new chat the user has opened but not yet sent into: no session, but a composer.
   * `null` when this is a saved session; otherwise a token that changes each time New is
   * clicked, which the composer watches to take the cursor back.
   */
  draft: number | null;
  onCreated: (id: string) => void;
}) {
  const qc = useQueryClient();
  const live = useRunStore(selectRun(sessionId));
  const ackMessages = useRunStore((s) => s.ackMessages);
  const messages = useQuery({
    queryKey: ["messages", sessionId],
    queryFn: () => api.messageList(sessionId!),
    enabled: sessionId != null,
  });
  const bottom = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  // Whether the view is parked at the end of the thread. A run streams text every few
  // frames, and following it down is right only while the user is actually reading the
  // end: scrolled up to re-read an earlier tool result, they must not be dragged back.
  // A ref and not state — nothing renders differently, and a scroll event per frame
  // should not cost a render.
  const atBottom = useRef(true);
  const onScroll = () => {
    const el = scroller.current;
    if (el) atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_SLACK;
  };
  // Another session starts at its own end, whatever the user had done to this one's.
  useEffect(() => {
    atBottom.current = true;
  }, [sessionId]);

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
    if (atBottom.current) bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.data?.length, live.text, live.thinking, live.toolOrder.length]);

  const { segments: stored, results, claimed } = useMemo(() => buildThread(messages.data ?? []), [messages.data]);

  // A session id can outlive the session: the one in `localStorage` after the row was
  // deleted, or one the list has not dropped yet. The engine says `not_found`; there is
  // nothing to compose into, so this reads as "no session" rather than as a failure.
  // A draft is empty on purpose and has a composer; "no session" is the state with neither.
  const gone = messages.isError && api.kindOf(messages.error) === "not_found";
  if ((!sessionId && draft === null) || gone) return <EmptyState text={S.noSession} />;

  const streaming = live.status === "running";
  // Calls the engine announced before it wrote the message that names them. Once that
  // message lands they belong to a stored group and this list empties.
  const unclaimed = streaming ? live.toolOrder.filter((id) => !claimed.has(id)) : [];
  const liveCalls: GroupCall[] = unclaimed.map((id) => {
    const c = live.toolCalls[id];
    return { id, name: c.name, args: c.arguments };
  });
  // Whether the live block has anything of its own. When it has, those calls came *after*
  // that text and stay under it in a group of their own; when it has not — the usual case,
  // because the engine writes the assistant message before running its tools and that
  // resets the stream — they fold onto the end of the last stored group instead of opening
  // a second one that merges into it a moment later.
  const saidLive = !!live.text || !!live.thinking;
  const segments = saidLive ? stored : withLiveCalls(stored, liveCalls);
  const liveKey = liveGroupKey(segments, streaming);
  // A run that has produced nothing yet still says so.
  const pulse = streaming && !live.text && !live.thinking && live.toolOrder.length === 0;
  const showLive = streaming && (pulse || saidLive);
  // A call as its card needs it: the live run where it has an entry, storage where it does
  // not, and `live` to tell a call still running from one nothing will ever answer.
  const resolve = (calls: GroupCall[], inFlight: boolean) =>
    calls.map((c) => resolveCall(c, live.toolCalls[c.id], results.get(c.id), inFlight));
  return (
    <>
      {/* `mask-fade-top` thins messages out as they pass under the title bar, which has no
          rule of its own to stop them at. The top padding is derived from the fade rather
          than picked: the mask dims everything within `--fade-top` of the edge, scrolled
          or not, so anything less would leave the first message greyed out at rest. */}
      <div
        ref={scroller}
        onScroll={onScroll}
        className="mask-fade-top min-h-0 flex-1 overflow-y-auto px-6 pt-[calc(var(--fade-top)+0.5rem)] pb-4"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.isError && (
            <div className="text-sm text-destructive">
              {S.loadFailed} ({api.messageOf(messages.error)})
            </div>
          )}
          {segments.map((seg) =>
            seg.kind === "turn" ? (
              seg.message.message.role === "user" ? (
                <UserBubble key={seg.key} message={seg.message.message} />
              ) : (
                <AssistantBubble key={seg.key} message={seg.message.message} />
              )
            ) : (
              <ToolGroup
                key={seg.key}
                calls={resolve(seg.calls, seg.key === liveKey)}
                thinking={seg.thinking}
                live={seg.key === liveKey}
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
              {liveCalls.length > 0 && <ToolGroup calls={resolve(liveCalls, true)} thinking={[]} live />}
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
      <Composer sessionId={sessionId} draft={draft} onCreated={onCreated} />
    </>
  );
}
