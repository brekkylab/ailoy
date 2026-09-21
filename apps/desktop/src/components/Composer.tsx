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
import { SendHorizontal, Square } from "lucide-react";
import { useState } from "react";

import * as api from "@/api";
import { UsageBar } from "@/components/UsageBar";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cancelRun, startRun } from "@/events";
import { hasAnyKey } from "@/lib/settings";
import { selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";

export function Composer({
  sessionId,
  onCreated,
}: {
  /** `null` on an unsaved new chat: the first send is what brings a session into being. */
  sessionId: string | null;
  /** Called with the new session's id once the first message has started its run. */
  onCreated: (id: string) => void;
}) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const live = useRunStore(selectRun(sessionId));
  const running = live.status === "running";
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList });
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
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

  return (
    <div className="border-t px-6 py-3">
      <div className="mx-auto max-w-3xl">
        {/* Nothing has been spent on a draft, and the bar is also what refetches a
            session's state when a run ends — neither applies until there is a session. */}
        {sessionId && <UsageBar sessionId={sessionId} />}
        <div className="flex items-end gap-2 rounded-xl border bg-background p-2">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder={noKey ? S.noKey : S.composerPlaceholder}
            rows={2}
            aria-label={S.messageInput}
            className="min-h-10 flex-1 resize-none border-0 shadow-none focus-visible:ring-0"
          />
          {running ? (
            <Button variant="destructive" size="icon" onClick={() => cancel.mutate()} aria-label={S.stop}>
              <Square className="size-4" />
            </Button>
          ) : (
            <Button size="icon" onClick={() => send.mutate()} disabled={!canSend} aria-label={S.send}>
              <SendHorizontal className="size-4" />
            </Button>
          )}
        </div>
        <div className="mt-1 flex items-center gap-3">
          <Select
            value={model}
            onValueChange={(m) => {
              if (!m) return;
              // A draft has no row to update, so its choice is just remembered until the
              // send that creates the session passes it to `session_create`.
              if (sessionId) setModel.mutate(m);
              else setDraftModel(m);
            }}
            disabled={running}
          >
            <SelectTrigger size="sm" className="w-72 text-xs" aria-label={S.model}>
              {/* Base UI takes the trigger's text from the selected *item*, and the items
                  live in a portal that only mounts once the list has been opened — so a
                  session restored from storage would show its bare model id until then.
                  Formatting from the catalog here is what makes the closed trigger read
                  the same as the open list. The lookup spans every model, not just the
                  available ones, so a session pinned to a model whose key was removed
                  still shows a name. */}
              <SelectValue>
                {(id: unknown) => {
                  if (typeof id !== "string" || !id) return S.model;
                  const m = models.data?.find((x) => x.id === id);
                  return m ? `${m.provider} · ${m.name}` : id;
                }}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(models.data ?? [])
                .filter((m) => m.available)
                .map((m) => (
                  <SelectItem key={m.id} value={m.id} className="text-xs">
                    {m.provider} · {m.name}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          {send.isError && <span className="text-xs text-destructive">{api.messageOf(send.error)}</span>}
          {setModel.isError && <span className="text-xs text-destructive">{api.messageOf(setModel.error)}</span>}
          {authFailed && <span className="text-xs text-destructive">{S.authFailed}</span>}
          {live.status === "error" && live.error?.kind === "max_turns" && (
            <Button size="sm" variant="outline" onClick={() => resume.mutate()} disabled={resume.isPending}>
              {S.continueRun}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
