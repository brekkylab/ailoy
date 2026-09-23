import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { confirm } from "@tauri-apps/plugin-dialog";
import { cn } from "cn";
import { FolderTree, MessageSquarePlus, MessagesSquare, Package, Pencil, Trash } from "lucide-react";
import { useEffect, useState } from "react";

import * as api from "@/api";
import { SourcesList } from "@/components/sources/SourcesList";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { groupSessions } from "@/lib/sessionGroups";
import { S } from "@/strings";
import type { MainView } from "@/views";

/** One row of the panel nav, styled to match a session row so the list reads as one column. */
function NavRow({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
        active && "bg-accent",
      )}
      aria-current={active ? "page" : undefined}
      onClick={onClick}
    >
      {icon}
      <span className="truncate">{label}</span>
    </button>
  );
}

export function Sidebar({
  selected,
  onSelect,
  onNewChat,
  drafting,
  view,
  onSelectView,
  source,
  onSelectSource,
}: {
  selected: string | null;
  /** `null` after the selected session is deleted: `App` then picks the next one. */
  onSelect: (id: string | null) => void;
  /** Opens an unsaved chat. Nothing is stored until the user sends into it. */
  onNewChat: () => void;
  /** Whether the thread on screen is that unsaved chat, which is `New chat`'s row to mark. */
  drafting: boolean;
  /** Which of the three the main panel is showing, so this can mark the active row. */
  view: MainView;
  onSelectView: (view: MainView) => void;
  /** The connected source the workspace panel is showing, `null` for the root. */
  source: string | null;
  onSelectSource: (path: string) => void;
}) {
  const qc = useQueryClient();
  // Polled: `running` drives the per-row dot, and a run started in another window — or
  // finished while this list sat idle — has no event of its own to invalidate on.
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList, refetchInterval: 5000 });
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // One instant for the whole list, advanced on its own clock. The query above refetches
  // every 5s but only re-renders when the rows actually change, so without this a session
  // would stay under Today past midnight for as long as nothing else happened in the app.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);
  // One line for both mutations: they are mutually exclusive in practice, and a rejected
  // rename or delete otherwise fails silently — the list just does not move. Creating is
  // not among them any more: New opens a draft, and the session is created by the first
  // send, where the composer already has somewhere to put the error.
  const [error, setError] = useState<string | null>(null);

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
    <aside className="flex h-full flex-col border-r bg-muted/60">
      {/* What the window can show. One row each, above the list below, because that list
          is unbounded — under it they would be the first thing to scroll away.

          `Chats` is a destination rather than the action `New chat` is, and it is what
          makes the list below safe to swap: with the workspace open the sessions are not
          on screen, and without a way back the only route to an existing conversation
          would be to start a new one.

          `New chat` heads the list as a row like the others rather than as a filled button:
          it is the most used action here, and it still does not need the one colour the
          window keeps for sending. */}
      <nav className="space-y-0.5 px-2 pt-2 pb-3">
        <NavRow
          icon={<MessageSquarePlus className="size-4" />}
          label={S.newChat}
          active={view === "session" && drafting}
          onClick={onNewChat}
        />
        <NavRow
          icon={<MessagesSquare className="size-4" />}
          label={S.chats}
          active={view === "session" && !drafting}
          onClick={() => onSelectView("session")}
        />
        <NavRow
          icon={<FolderTree className="size-4" />}
          label={S.workspace}
          active={view === "workspace"}
          onClick={() => onSelectView("workspace")}
        />
        <NavRow
          icon={<Package className="size-4" />}
          label={S.artifacts}
          active={view === "artifacts"}
          onClick={() => onSelectView("artifacts")}
        />
      </nav>
      {/* The lower half belongs to whichever panel is open: the sources that feed the
          workspace while it is showing, the conversations otherwise. */}
      {view === "workspace" ? (
        <SourcesList selected={source} onSelect={onSelectSource} />
      ) : (
        <>
      {error && <p className="px-3 pb-2 text-xs text-destructive">{error}</p>}
      <ScrollArea className="min-h-0 flex-1 px-2">
        {/* Under headed stretches of time rather than with a date on every row — see
            `groupSessions`. A row is its title alone: the model is the composer's to show,
            and still on the row's tooltip. */}
        {groupSessions(sessions.data ?? [], now).map((group) => (
          <section key={group.label} className="pb-3">
            <h3 className="px-2 pb-1 text-xs font-medium text-muted-foreground">{group.label}</h3>
            {group.sessions.map((s) => (
              <div
                key={s.id}
                className={cn(
                  "group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
                  // Only while the thread is what the main panel is actually showing: a
                  // highlighted row next to an open workspace would claim the window.
                  view === "session" && selected === s.id && "bg-accent",
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
                    <button className="min-w-0 flex-1 text-left" onClick={() => onSelect(s.id)} title={`${s.title}\n${s.model}`}>
                      <div className="truncate">
                        {s.running && (
                          <>
                            {/* The dot is the only thing that says a session is working; a
                                screen reader gets the word instead of a bare bullet. */}
                            <span className="sr-only">{S.running}</span>
                            <span className="mr-1 inline-block size-2 animate-pulse rounded-full bg-emerald-500" />
                          </>
                        )}
                        {s.title}
                      </div>
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
          </section>
        ))}
        {/* A list that failed to load is not an empty list: saying "Empty" there would
            invite the user to start typing into a storage layer that is not answering. */}
        {sessions.isError ? (
          <p className="p-3 text-xs text-destructive">{api.messageOf(sessions.error)}</p>
        ) : (
          sessions.data?.length === 0 && <p className="p-3 text-xs text-muted-foreground">{S.empty}</p>
        )}
      </ScrollArea>
        </>
      )}
    </aside>
  );
}
