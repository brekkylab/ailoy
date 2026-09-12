import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import * as api from "@/api";
import { Banner } from "@/components/Banner";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Sidebar } from "@/components/Sidebar";
import { Thread } from "@/components/Thread";
import { WorkspacePanel } from "@/components/WorkspacePanel";
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
  useEffect(() => {
    try {
      if (selected) localStorage.setItem(KEY, selected);
    } catch {
      /* storage may be unavailable */
    }
  }, [selected]);
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const noKey = settings.data && !settings.data.providers.some((p) => p.has_key);

  return (
    <div className="grid h-full grid-cols-[260px_minmax(0,1fr)_320px]">
      <Sidebar selected={selected} onSelect={setSelected} onOpenSettings={() => setSettingsOpen(true)} />
      <main className="flex h-full min-w-0 flex-col">
        {ws.data?.status.status === "degraded" && <Banner text={`${S.degraded} (${ws.data.status.reason})`} />}
        {noKey && <Banner text={S.noKey} tone="error" />}
        <Thread sessionId={selected} />
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
