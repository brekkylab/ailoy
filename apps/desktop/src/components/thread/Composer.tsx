// The composer: what the user types, the model they type it at, and the one button that
// is either "send" or "stop".
//
// Sending is a mutation over `startRun`, not a fire-and-forget call, so that the engine
// rejecting the start (`already_running`, a model that vanished from the catalog) has
// somewhere to land. Everything about the run *after* it starts is the event stream's
// business and is read back out of the run store — this component only starts and
// cancels.
//
// `sessionId` is null while the window is on a new, unsaved chat. Nothing is stored for
// one of those until there is a message to store: clicking New used to write a row, which
// left a drift of empty "New chat" sessions behind every time someone opened one and
// changed their mind. So the draft keeps its model in local state, and the first send
// creates the session and starts the run as one action — the two together, because a
// session that exists without the message that caused it is the thing being avoided.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import * as api from "@/api";
import { ModelPicker } from "@/components/thread/ModelPicker";
import { UsageBar } from "@/components/thread/UsageBar";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cancelRun, startRun } from "@/events";
import { catalogQuery, useRefreshModels } from "@/lib/catalog";
import { hasAnyKey } from "@/lib/settings";
import { selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";

export function Composer({
  sessionId,
  draft,
  onCreated,
}: {
  /** `null` on an unsaved new chat: the first send is what brings a session into being. */
  sessionId: string | null;
  /** The open draft's token, or `null` for a saved session. Only the focus below reads it. */
  draft: number | null;
  /** Called with the new session's id once the first message has started its run. */
  onCreated: (id: string) => void;
}) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);
  // Opening a new chat puts the cursor where the user is about to type. Keyed on the draft
  // token as well as the session, so clicking New a second time brings focus back from the
  // button it just landed on — without it, nothing about the window would have changed and
  // the effect would not run.
  useEffect(() => {
    if (sessionId === null) box.current?.focus();
  }, [sessionId, draft]);
  const live = useRunStore(selectRun(sessionId));
  const running = live.status === "running";
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  // An empty list is a first start that has not reached models.dev yet: the picker would
  // open onto nothing, so the row says which of the two it is waiting on.
  const catalog = useQuery(catalogQuery);
  const refreshModels = useRefreshModels();
  const noModels = catalog.data?.models === 0;
  const session = sessions.data?.find((s) => s.id === sessionId);
  // No key anywhere means the engine cannot build a model and would fail the run at
  // `build()`; say so in the place the user is about to type instead.
  const noKey = !hasAnyKey(settings.data);

  // A draft's model has nowhere to be stored yet, so it lives here until the session does.
  // Null means "whatever the settings default is", which is also what `session_create`
  // does with no model, so an untouched draft and the engine agree without being told.
  const [draftModel, setDraftModel] = useState<string | null>(null);
  const model = sessionId ? (session?.model ?? null) : (draftModel ?? settings.data?.default_model ?? null);

  const setModel = useMutation({
    mutationFn: (m: string) => api.sessionSetModel(sessionId!, m),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
  // A cancel that the engine rejects — the run ended a moment ago, the session is gone —
  // is not worth a message, but it is worth catching: an unhandled rejection in the
  // webview is a console error the user cannot act on.
  const cancel = useMutation({ mutationFn: () => cancelRun(sessionId!) });
  // `startRun` resolves as soon as the engine accepts the run; the session's `running`
  // flag is what the sidebar paints, hence the invalidate here and not on the terminal
  // event (`UsageBar` owns that one).
  //
  // On a draft both steps are in the one mutation, and the window is told about the new
  // session only after the run is away. Switching earlier would unmount this composer
  // mid-request and take the error with it — which is the one case where the user most
  // needs to see what went wrong, since their message is in the box that vanished.
  const send = useMutation({
    // Resolves to the session this send had to create, or null when it sent into one that
    // already existed. Not the run id: `startRun` returns that, and the two are both bare
    // strings, so handing back whichever came last would read fine and mean nothing.
    mutationFn: async (): Promise<string | null> => {
      const body = text.trim();
      if (sessionId) {
        await startRun(sessionId, body);
        return null;
      }
      const created = await api.sessionCreate(model ?? undefined);
      await startRun(created.id, body);
      return created.id;
    },
    onSuccess: (createdId) => {
      setText("");
      void qc.invalidateQueries({ queryKey: ["sessions"] });
      if (createdId) onCreated(createdId);
    },
  });
  // The engine stops a run at `max_turns` rather than looping forever; "continue" is the
  // message that resumes it, and is a model-facing payload, not UI copy.
  const resume = useMutation({
    mutationFn: () => startRun(sessionId!, "continue"),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["sessions"] }),
  });

  const canSend = text.trim().length > 0 && !running && !noKey && !send.isPending;
  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // `isComposing` is the whole reason this is `keydown` and not `keypress`: an IME
    // commits a Hangul syllable with Enter, and sending there would eat the character.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (canSend) send.mutate();
    }
  };

  const authFailed = live.status === "error" && live.error?.kind === "model" && live.error.status === 401;

  const noticeRow =
    noModels || send.isError || setModel.isError || authFailed || (live.status === "error" && live.error?.kind === "max_turns");

  return (
    <div className="px-6 pt-2 pb-4">
      <div className="mx-auto max-w-3xl">
        {/* One card, the way Claude and ChatGPT draw it: the box you type in, and under it the
            row of what the message will be sent with — the model on the left, the context
            it has used and the send button on the right. The textarea wears no field chrome
            of its own; the card is the field, and it is what lights up on focus. */}
        <div className="rounded-2xl border bg-card shadow-sm transition-colors focus-within:border-ring/60">
          <Textarea
            ref={box}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder={noKey ? S.noKey : S.composerPlaceholder}
            rows={1}
            aria-label={S.messageInput}
            className="max-h-60 min-h-12 resize-none border-0 bg-transparent px-4 pt-3.5 pb-1 text-[15px] shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
          <div className="flex items-center gap-1 px-2 pb-2">
            <ModelPicker
              value={model}
              onChange={(m) => {
                // A draft has no row to update, so its choice is just remembered until the
                // send that creates the session passes it to `session_create`.
                if (sessionId) setModel.mutate(m);
                else setDraftModel(m);
              }}
              disabled={running}
            />
            <div className="ml-auto flex items-center gap-2">
              {/* Nothing has been spent on a draft, and the bar is also what refetches a
                  session's state when a run ends — neither applies until there is a session. */}
              {sessionId && <UsageBar sessionId={sessionId} />}
              {running ? (
                <Button size="icon" className="size-8 rounded-full" onClick={() => cancel.mutate()} aria-label={S.stop}>
                  <Square className="size-3.5 fill-current" />
                </Button>
              ) : (
                <Button
                  size="icon"
                  className="size-8 rounded-full"
                  onClick={() => send.mutate()}
                  disabled={!canSend}
                  aria-label={S.send}
                >
                  <ArrowUp className="size-4" />
                </Button>
              )}
            </div>
          </div>
        </div>
        {noticeRow && (
          <div className="mt-1.5 flex flex-wrap items-center gap-3 px-2">
          {noModels &&
            (catalog.data?.error && !catalog.data.refreshing ? (
              <>
                <span className="text-xs text-destructive" title={catalog.data.error}>
                  {S.modelsLoadFailed}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => refreshModels.mutate()}
                  disabled={refreshModels.isPending}
                >
                  {S.retry}
                </Button>
              </>
            ) : (
              <span className="text-xs text-muted-foreground">{S.modelsLoading}</span>
            ))}
          {send.isError && <span className="text-xs text-destructive">{api.messageOf(send.error)}</span>}
          {setModel.isError && <span className="text-xs text-destructive">{api.messageOf(setModel.error)}</span>}
          {authFailed && <span className="text-xs text-destructive">{S.authFailed}</span>}
          {live.status === "error" && live.error?.kind === "max_turns" && (
            <Button size="sm" variant="outline" onClick={() => resume.mutate()} disabled={resume.isPending}>
              {S.continueRun}
            </Button>
          )}
          </div>
        )}
        {sessionId === null && !text && (
          // A draft is a blank page; these are a way onto it. Picking one only fills the box —
          // the user still reads it and sends it, because a starter that ran on click would
          // spend a model call on a guess.
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {S.starters.map((starter) => (
              <button
                key={starter}
                className="rounded-full border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                onClick={() => {
                  setText(starter);
                  box.current?.focus();
                }}
              >
                {starter}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
