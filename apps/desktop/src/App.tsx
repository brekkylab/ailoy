import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import * as api from "@/api";
import { Banner } from "@/components/Banner";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Sidebar } from "@/components/Sidebar";
import { Thread } from "@/components/Thread";
import { WorkspacePanel } from "@/components/WorkspacePanel";
import { hasAnyKey } from "@/lib/settings";
import { S } from "@/strings";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });
const KEY = "ailoy.session";

/** The generated tokens carry a `.dark` set; nothing else toggles it, so the OS does. */
function useSystemTheme() {
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => document.documentElement.classList.toggle("dark", mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
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
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: api.sessionList });
  // `selected` is the user's explicit choice; `effective` is what the window shows. Derived
  // during render rather than written back in an effect: the sessions cache is one refetch
  // behind whatever action just changed it, so an effect that "corrected" the selection
  // against the stale list would revert a freshly created session to the old head and
  // briefly show a just-deleted one. A derived value falls back to the most recently
  // updated session (`session_list` is ordered `updated_at DESC`) only while the choice
  // is unknown or gone, and snaps back to the choice the moment the list catches up.
  const list = sessions.data;
  const effective =
    selected && (!list || list.some((s) => s.id === selected)) ? selected : (list?.[0]?.id ?? null);
  useEffect(() => {
    try {
      if (effective) localStorage.setItem(KEY, effective);
      else localStorage.removeItem(KEY);
    } catch {
      /* storage may be unavailable */
    }
  }, [effective]);
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const noKey = !hasAnyKey(settings.data);

  return (
    <div className="grid h-full grid-rows-[minmax(0,1fr)] grid-cols-[260px_minmax(0,1fr)_320px]">
      <Sidebar selected={effective} onSelect={setSelected} onOpenSettings={() => setSettingsOpen(true)} />
      <main className="flex h-full min-w-0 flex-col">
        {ws.data?.status.status === "degraded" && <Banner text={`${S.degraded} (${ws.data.status.reason})`} />}
        {noKey && <Banner text={S.noKey} tone="error" />}
        <Thread sessionId={effective} />
      </main>
      <WorkspacePanel />
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
