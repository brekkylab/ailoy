import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { confirm } from "@tauri-apps/plugin-dialog";
import { cn } from "cn";
import { MessageSquarePlus, Pencil, Settings as SettingsIcon, Trash } from "lucide-react";
import { useState } from "react";

import * as api from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { S } from "@/strings";

export function Sidebar({
  selected,
  onSelect,
  onOpenSettings,
}: {
  selected: string | null;
  /** `null` after the selected session is deleted: `App` then picks the next one. */
  onSelect: (id: string | null) => void;
  onOpenSettings: () => void;
}) {
  const qc = useQueryClient();
  // Polled: `running` drives the per-row dot, and a run started in another window — or
  // finished while this list sat idle — has no event of its own to invalidate on.
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList, refetchInterval: 5000 });
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // One line for all three mutations: they are mutually exclusive in practice, and a
  // rejected create/rename/delete otherwise fails silently — the list just does not move.
  const [error, setError] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => api.sessionCreate(),
    onSuccess: (s) => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["sessions"] });
      onSelect(s.id);
    },
    onError: (err) => setError(api.messageOf(err)),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.sessionDelete(id),
    onSuccess: (_void, id) => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["sessions"] });
      // Deselect rather than guess: `App` owns which session takes over.
      if (id === selected) onSelect(null);
    },
    onError: (err) => setError(api.messageOf(err)),
  });
  const rename = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.sessionRename(id, title),
    onSuccess: () => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["sessions"] });
    },
    onError: (err) => setError(api.messageOf(err)),
  });

  // WKWebView implements neither `window.prompt` nor a reliable `window.confirm`, so the
  // rename edits in place and the delete asks through the Tauri dialog plugin.
  const commitRename = (id: string, current: string) => {
    const title = draft.trim();
    setEditing(null);
    if (title && title !== current) rename.mutate({ id, title });
  };
  const askDelete = async (id: string) => {
    if (await confirm(S.confirmDelete, { title: S.delete, kind: "warning" })) remove.mutate(id);
  };

  const rowAction = "rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100";

  return (
    <aside className="flex h-full flex-col border-r bg-muted/30">
      <div className="flex items-center gap-2 p-3">
        <Button className="flex-1" onClick={() => create.mutate()} disabled={create.isPending}>
          <MessageSquarePlus className="size-4" /> {S.newChat}
        </Button>
        <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label={S.settings}>
          <SettingsIcon className="size-4" />
        </Button>
      </div>
      {error && <p className="px-3 pb-2 text-xs text-destructive">{error}</p>}
      <ScrollArea className="flex-1 px-2">
        {(sessions.data ?? []).map((s) => (
          <div
            key={s.id}
            className={cn(
              "group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
              selected === s.id && "bg-accent",
            )}
          >
            {editing === s.id ? (
              <Input
                autoFocus
                aria-label={S.rename}
                className="h-6 flex-1"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitRename(s.id, s.title);
                  else if (e.key === "Escape") setEditing(null);
                }}
                onBlur={() => setEditing(null)}
              />
            ) : (
              <>
                <button className="flex-1 truncate text-left" onClick={() => onSelect(s.id)} title={s.model}>
                  {s.running && <span className="mr-1 inline-block size-2 animate-pulse rounded-full bg-emerald-500" />}
                  {s.title}
                </button>
                <button
                  className={rowAction}
                  aria-label={S.rename}
                  onClick={() => {
                    setDraft(s.title);
                    setEditing(s.id);
                  }}
                >
                  <Pencil className="size-3.5" />
                </button>
                <button className={rowAction} aria-label={S.delete} onClick={() => void askDelete(s.id)}>
                  <Trash className="size-3.5" />
                </button>
              </>
            )}
          </div>
        ))}
        {sessions.data?.length === 0 && <p className="p-3 text-xs text-muted-foreground">{S.empty}</p>}
      </ScrollArea>
    </aside>
  );
}
