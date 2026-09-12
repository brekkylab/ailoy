// The composer: what the user types, the model they type it at, and the one button that
// is either "send" or "stop".
//
// Sending is a mutation over `startRun`, not a fire-and-forget call, so that the engine
// rejecting the start (`already_running`, a model that vanished from the catalog) has
// somewhere to land. Everything about the run *after* it starts is the event stream's
// business and is read back out of the run store — this component only starts and
// cancels.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { SendHorizontal, Square } from "lucide-react";
import { useState } from "react";

import * as api from "@/api";
import { UsageBar } from "@/components/UsageBar";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cancelRun, startRun } from "@/events";
import { selectRun, useRunStore } from "@/store/runs";
import { S } from "@/strings";

export function Composer({ sessionId }: { sessionId: string }) {
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
  const noKey = settings.data != null && settings.data.providers.every((p) => !p.has_key);

  const setModel = useMutation({
    mutationFn: (m: string) => api.sessionSetModel(sessionId, m),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
  // `startRun` resolves as soon as the engine accepts the run; the session's `running`
  // flag is what the sidebar paints, hence the invalidate here and not on the terminal
  // event (`UsageBar` owns that one).
  const send = useMutation({
    mutationFn: () => startRun(sessionId, text.trim()),
    onSuccess: () => {
      setText("");
      void qc.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
  // The engine stops a run at `max_turns` rather than looping forever; "continue" is the
  // message that resumes it, and is a model-facing payload, not UI copy.
  const resume = useMutation({
    mutationFn: () => startRun(sessionId, "continue"),
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
        <UsageBar sessionId={sessionId} />
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
            <Button variant="destructive" size="icon" onClick={() => void cancelRun(sessionId)} aria-label={S.stop}>
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
            value={session?.model ?? null}
            onValueChange={(m) => {
              if (m) setModel.mutate(m);
            }}
            disabled={running}
          >
            <SelectTrigger size="sm" className="w-72 text-xs" aria-label={S.model}>
              <SelectValue placeholder={session?.model ?? S.model} />
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
