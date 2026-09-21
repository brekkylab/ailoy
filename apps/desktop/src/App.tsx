import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

import * as api from "@/api";
import { ArtifactsPanel } from "@/components/ArtifactsPanel";
import { Banner } from "@/components/Banner";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Sidebar } from "@/components/Sidebar";
import { Thread } from "@/components/Thread";
import { TitleBar } from "@/components/TitleBar";
import { WorkspacePanel } from "@/components/WorkspacePanel";
import { sessionTitle } from "@/lib/sessionTitle";
import { hasAnyKey } from "@/lib/settings";
import { S } from "@/strings";
import type { MainView } from "@/views";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });
const KEY = "ailoy.session";
const SIDEBAR_KEY = "ailoy.sidebarCollapsed";
/** One source for the sidebar's width: the grid column and the title bar's left segment. */
const SIDEBAR_VARS = { "--sidebar-w": "260px" } as CSSProperties;

/**
 * Keeps Tailwind's `.dark` class on the same side as the palette.
 *
 * The colours come from `light-dark()` pairs and need no help — CSS resolves them off
 * `color-scheme`. This is only for the handful of `dark:` variants in the components,
 * which key off a class instead and would otherwise be reading a different theme from
 * everything around them.
 *
 * `data-theme` wins where it is set, matching the three `color-scheme` rules in
 * `index.css`, so a theme picker added later moves both halves by setting that one
 * attribute. Absent it, the OS decides and the listener follows it switching at night.
 */
function useSystemTheme() {
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const root = document.documentElement;
    const apply = () => {
      const forced = root.dataset.theme;
      root.classList.toggle("dark", forced ? forced === "dark" : mq.matches);
    };
    apply();
    mq.addEventListener("change", apply);
    const observer = new MutationObserver(apply);
    observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", apply);
      observer.disconnect();
    };
  }, []);
}

function Shell() {
  const [selected, setSelected] = useState<string | null>(() => {
    try {
      return localStorage.getItem(KEY);
    } catch {
      return null;
    }
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Not persisted, unlike the session and the sidebar: a relaunch should land on the
  // conversation, which is what the window is for.
  const [view, setView] = useState<MainView>("session");
  // A new chat that has not been sent into yet. It holds the window on an empty thread
  // without anything being stored, which is the point: a session is written by the first
  // message, not by the click that opened the window for it.
  // A token rather than a flag: it changes on every New click, which is what lets the
  // composer put the cursor back in the box when the second click left it on the button
  // and nothing else about the window moved.
  const [draft, setDraft] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_KEY) === "1";
    } catch {
      return false;
    }
  });
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList });
  // `selected` is the user's explicit choice; `effective` is what the window shows. Derived
  // during render rather than written back in an effect: the sessions cache is one refetch
  // behind whatever action just changed it, so an effect that "corrected" the selection
  // against the stale list would revert a freshly created session to the old head and
  // briefly show a just-deleted one. A derived value falls back to the most recently
  // updated session (`session_list` is ordered `updated_at DESC`) only while the choice
  // is unknown or gone, and snaps back to the choice the moment the list catches up.
  //
  // A draft overrides all of it: the window is showing a chat that has no id yet, and the
  // fallback to the newest session would otherwise drag it straight back to the one the
  // user just clicked away from.
  const list = sessions.data;
  const stored =
    selected && (!list || list.some((s) => s.id === selected)) ? selected : (list?.[0]?.id ?? null);
  const effective = draft !== null ? null : stored;
  // Follows `stored`, not `effective`: a draft is not a session and must not erase which
  // one the window would go back to, or opening a new chat and closing the app would lose
  // the conversation that was open before it.
  useEffect(() => {
    try {
      if (stored) localStorage.setItem(KEY, stored);
      else localStorage.removeItem(KEY);
    } catch {
      /* storage may be unavailable */
    }
  }, [stored]);
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const noKey = !hasAnyKey(settings.data);
  // Only a conversation is named up here. The other two panels carry their own heading,
  // where there is room to set it larger than a title bar allows.
  const title = view === "session" ? sessionTitle(effective, list) : null;

  // Picking a session is also how you get back out of the workspace and artifacts views,
  // and out of an unsent draft: there is no other way back, and a click on a conversation
  // can only mean show it.
  const selectSession = (id: string | null) => {
    setSelected(id);
    setDraft(null);
    setView("session");
  };
  const newChat = () => {
    setDraft((n) => (n ?? 0) + 1);
    setView("session");
  };

  const toggleSidebar = () =>
    setCollapsed((c) => {
      const next = !c;
      try {
        localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
      } catch {
        /* storage may be unavailable */
      }
      return next;
    });

  return (
    <div
      style={SIDEBAR_VARS}
      className={cn(
        "grid h-full grid-rows-[auto_minmax(0,1fr)]",
        collapsed ? "grid-cols-[minmax(0,1fr)]" : "grid-cols-[var(--sidebar-w)_minmax(0,1fr)]",
      )}
    >
      <TitleBar
        title={title}
        collapsed={collapsed}
        onToggleSidebar={toggleSidebar}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      {/* Unmounted rather than hidden when collapsed: a sidebar of zero width still takes
          tab stops, and its session list would keep polling behind the fold. */}
      {!collapsed && (
        <Sidebar
          selected={effective}
          onSelect={selectSession}
          onNewChat={newChat}
          view={view}
          onSelectView={setView}
        />
      )}
      <main className="flex h-full min-w-0 flex-col">
        {/* Both are conditions on the app rather than on the conversation, and the mount
            one is at its most relevant on the workspace view, so they sit above whichever
            panel is open. */}
        {ws.data?.status.status === "degraded" && <Banner text={`${S.degraded} (${ws.data.status.reason})`} />}
        {noKey && <Banner text={S.noKey} tone="error" />}
        {view === "session" && (
          <Thread sessionId={effective} draft={draft} onCreated={selectSession} />
        )}
        {view === "workspace" && <WorkspacePanel />}
        {view === "artifacts" && <ArtifactsPanel />}
      </main>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}

export default function App() {
  useSystemTheme();
  return (
    <QueryClientProvider client={qc}>
      <Shell />
    </QueryClientProvider>
  );
}
