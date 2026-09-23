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
import { cn } from "cn";
import { useEffect, useMemo, useRef } from "react";

import * as api from "@/api";
import { Markdown } from "@/components/Markdown";
import { Composer } from "@/components/thread/Composer";
import { AssistantBubble, UserBubble } from "@/components/thread/MessageBubble";
import { ToolGroup } from "@/components/thread/ToolGroup";
import { attachRun } from "@/events";
import { buildThread, liveGroupKey, resolveCall, turnEnds, withLiveCalls, type GroupCall } from "@/lib/thread";
import { ENTER, useEntrance, type EntranceRow } from "@/lib/useEntrance";
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
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  // Whether the view is parked at the end of the thread. A run streams text every few
  // frames, and following it down is right only while the user is actually reading the
  // end: scrolled up to re-read an earlier tool result, they must not be dragged back.
  // A ref and not state — nothing renders differently, and a scroll event per frame
  // should not cost a render.
  const atBottom = useRef(true);
  // Where the last scroll event left the view, to tell which way the next one went.
  const lastTop = useRef(0);
  // Only the reader lets go of the end, by scrolling up. Reading the distance from the end
  // alone was the first version, and it let go by itself: a scroll this component makes is
  // reported a frame later, and a long answer that lays out a piece per frame has grown by
  // then — so its own scroll to the end read as one that had stopped short, and the view
  // stayed where the answer started. Growth never moves `scrollTop` up; the reader does.
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (gap <= 2) atBottom.current = true;
    else if (el.scrollTop < lastTop.current) atBottom.current = false;
    else if (gap < BOTTOM_SLACK) atBottom.current = true;
    lastTop.current = el.scrollTop;
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

  // Follow the end of the thread for as long as the reader is at it — keyed on the size of
  // what is on screen, not on the data that fills it. The data was the first version, and it
  // missed most of what makes a thread taller: a long answer's markdown lays itself out a
  // piece per frame after the render that delivered it, code is highlighted a frame or two
  // later, a tool call's row grows a preview and then a result, and the stored message
  // takes over from the streamed one at a height of its own. A resize is all of them.
  //
  // `scrollTop` rather than `scrollIntoView`, which would also scroll every ancestor that
  // can — the window included.
  useEffect(() => {
    const box = content.current;
    const sc = scroller.current;
    if (!box || !sc) return;
    const follow = () => {
      if (atBottom.current) sc.scrollTop = sc.scrollHeight;
    };
    follow();
    const observer = new ResizeObserver(follow);
    observer.observe(box);
    return () => observer.disconnect();
  }, [sessionId]);

  // Sending is asking to see the answer: a new run takes the view to the end even from
  // wherever the reader had scrolled up to, and it follows from there.
  useEffect(() => {
    if (!live.runId) return;
    atBottom.current = true;
    const sc = scroller.current;
    if (sc) sc.scrollTop = sc.scrollHeight;
  }, [live.runId]);

  const { segments: stored, results, claimed } = useMemo(() => buildThread(messages.data ?? []), [messages.data]);

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
  // Where each finished answer's copy button and time go. The answer a run is still
  // writing gets none until it ends.
  const ends = turnEnds(segments, streaming);
  // A run that has produced nothing yet still says so.
  const pulse = streaming && !live.text && !live.thinking && live.toolOrder.length === 0;
  const showLive = streaming && (pulse || saidLive);

  // Which rows fade in: everything that is on screen, the block a run streams into
  // included — keyed on the model's stored messages, so each turn's block is new when it
  // opens. Not on the user's: the run's first block opens before the message that started it
  // is stored, and counting that message would open the same block twice. See `useEntrance`,
  // including why a stored answer taking over from the stream does not fade in.
  const modelTurns = (messages.data ?? []).filter((m) => m.message.role !== "user").length;
  const liveRow = `live:${modelTurns}`;
  const rows: EntranceRow[] = segments.map((seg) => ({
    key: seg.key,
    answer: seg.kind === "turn" && seg.message.message.role === "assistant",
  }));
  if (showLive) rows.push({ key: liveRow });
  const entering = useEntrance(sessionId, messages.isSuccess, rows, showLive && saidLive);

  // A session id can outlive the session: the one in `localStorage` after the row was
  // deleted, or one the list has not dropped yet. The engine says `not_found`; there is
  // nothing to compose into, so this reads as "no session" rather than as a failure.
  // A draft is empty on purpose and has a composer; "no session" is the state with neither.
  const gone = messages.isError && api.kindOf(messages.error) === "not_found";
  if ((!sessionId && draft === null) || gone) return <EmptyState text={S.noSession} />;
  // A draft has nothing to scroll, so the box is the page: the greeting and the composer
  // sit together in the middle of it, rather than an empty column over a box at the foot.
  // Sending creates the session and the thread takes the usual shape, composer at the end.
  if (!sessionId) {
    return (
      <div className="flex min-h-0 flex-1 flex-col justify-center pb-[12vh]">
        <h2 className="mb-6 text-center text-[26px] font-semibold tracking-tight">{S.greeting}</h2>
        <Composer sessionId={null} draft={draft} onCreated={onCreated} />
      </div>
    );
  }

  // A call as its card needs it: the live run where it has an entry, storage where it does
  // not, and `live` to tell a call still running from one nothing will ever answer.
  const resolve = (calls: GroupCall[], inFlight: boolean) =>
    calls.map((c) => resolveCall(c, live.toolCalls[c.id], results.get(c.id), inFlight));
  return (
    <>
      {/* `mask-fade-y` thins messages out as they pass under the title bar and into the
          composer, neither of which has a rule of its own to stop them at. The padding at
          each end is derived from its fade rather than picked: the mask dims everything
          within that distance of the edge, scrolled or not, so anything less would leave the
          first message — or the last — greyed out at rest. */}
      <div
        ref={scroller}
        onScroll={onScroll}
        className="mask-fade-y min-h-0 flex-1 overflow-y-auto px-6 pt-[calc(var(--fade-top)+0.5rem)] pb-[calc(var(--fade-bottom)+0.5rem)]"
      >
        {/* Tight between the steps of one answer — a line, its tool calls, the reply — and
            wider where a new turn begins, which is the user's bubble's own margin. */}
        <div ref={content} className="mx-auto flex max-w-3xl flex-col gap-2">
          {messages.isError && (
            <div className="text-sm text-destructive">
              {S.loadFailed} ({api.messageOf(messages.error)})
            </div>
          )}
          {segments.map((seg, i) => {
            const user = seg.kind === "turn" && seg.message.message.role === "user";
            return (
              <div
                key={seg.key}
                className={cn(
                  // A new turn begins with the user's message, and gets the room that says so.
                  user && i > 0 && "mt-4",
                  entering(rows[i]) && ENTER,
                )}
              >
                {seg.kind === "turn" ? (
                  user ? (
                    <UserBubble message={seg.message.message} at={seg.message.created_at} />
                  ) : (
                    <AssistantBubble message={seg.message.message} end={ends.get(seg.key)} />
                  )
                ) : (
                  <ToolGroup
                    calls={resolve(seg.calls, seg.key === liveKey)}
                    thinking={seg.thinking}
                    live={seg.key === liveKey}
                  />
                )}
              </div>
            );
          })}
          {showLive && (
            <div key={liveRow} className={cn("max-w-[92%] space-y-1", entering({ key: liveRow }) && ENTER)}>
              {live.thinking && (
                <pre className="max-h-32 overflow-auto rounded bg-muted/40 p-2 text-xs whitespace-pre-wrap text-muted-foreground">
                  {live.thinking}
                </pre>
              )}
              {live.text && <Markdown text={live.text} streaming />}
              {liveCalls.length > 0 && <ToolGroup calls={resolve(liveCalls, true)} thinking={[]} live />}
              {pulse && <span className="animate-pulse text-sm text-muted-foreground">…</span>}
            </div>
          )}
          {live.status === "error" && live.error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive">
              {S.errorPrefix} ({live.error.kind}): {live.error.message}
            </div>
          )}
        </div>
      </div>
      <Composer sessionId={sessionId} draft={draft} onCreated={onCreated} />
    </>
  );
}
