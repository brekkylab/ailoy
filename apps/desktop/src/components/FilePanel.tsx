// The files beside a conversation.
//
// The workspace view is a place of its own, and going there meant leaving the thread —
// so a file the agent had just written, or one the user wanted to ask about, was never on
// screen at the same time as the conversation about it. This is the same browser, as a
// column to the right of the thread: the workspace's sources under Files, what the agent
// made under Artifacts. A tool call's row opens its file here (`store/panel`), and a file
// open here can be put into the message being written ("Ask about this file").

import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { X } from "lucide-react";

import * as api from "@/api";
import { FileBrowser } from "@/components/FileBrowser";
import { SourceIcon } from "@/components/icons/SourceIcon";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SourceBrowser } from "@/components/WorkspacePanel";
import { ARTIFACTS_ROOT, WORKSPACE_ROOT } from "@/paths";
import { useComposerInbox } from "@/store/composer";
import { usePanel } from "@/store/panel";
import { S } from "@/strings";

/** How a file is named in a message: its workspace path, as the agent will read it. */
const mention = (path: string) => `\`${path}\``;

export function FilePanel() {
  const { tab, source, file, setTab, setSource, setFile, close } = usePanel();
  const insert = useComposerInbox((s) => s.insert);
  const mounts = useQuery({ queryKey: ["mounts"], queryFn: api.mountList });
  const sources = mounts.data ?? [];
  const current = sources.find((m) => m.path === (source ?? WORKSPACE_ROOT));
  const onAsk = (path: string) => insert(mention(path));

  const tabButton = (id: "files" | "artifacts", label: string) => (
    <button
      role="tab"
      aria-selected={tab === id}
      onClick={() => setTab(id)}
      className={cn(
        "rounded-md px-2.5 py-1 text-xs transition-colors",
        tab === id ? "bg-accent font-medium text-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
    </button>
  );

  return (
    <aside aria-label={S.filePanel} className="flex min-h-0 w-[min(32rem,42%)] shrink-0 flex-col border-l">
      <div className="flex shrink-0 items-center gap-1 px-2 py-1.5">
        <div role="tablist" className="flex items-center gap-0.5">
          {tabButton("files", S.files)}
          {tabButton("artifacts", S.artifacts)}
        </div>
        {tab === "files" && sources.length > 1 && (
          // Which source, when there is more than My Computer: the same list the sidebar
          // shows while the workspace view is open.
          <Select value={source ?? WORKSPACE_ROOT} onValueChange={(v) => v && setSource(v === WORKSPACE_ROOT ? null : v)}>
            <SelectTrigger
              size="sm"
              aria-label={S.mounts}
              className="ml-1 max-w-48 border-transparent bg-transparent text-xs shadow-none hover:bg-accent dark:bg-transparent dark:hover:bg-accent"
            >
              <SelectValue>
                {() =>
                  current ? (
                    <span className="flex min-w-0 items-center gap-1.5">
                      <SourceIcon kind={current.kind} root={current.path === WORKSPACE_ROOT} className="size-3.5 shrink-0" />
                      <span className="truncate">{current.label}</span>
                    </span>
                  ) : (
                    S.workspace
                  )
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {sources.map((m) => (
                <SelectItem key={m.path} value={m.path} className="text-xs">
                  <SourceIcon kind={m.kind} root={m.path === WORKSPACE_ROOT} className="size-3.5 shrink-0" />
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <button
          className="ml-auto rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
          onClick={close}
          aria-label={S.close}
          title={S.close}
        >
          <X className="size-3.5" />
        </button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col border-t">
        {tab === "files" ? (
          <SourceBrowser key={source ?? WORKSPACE_ROOT} source={source} layout="stacked" open={file} onOpenChange={setFile} onAsk={onAsk} />
        ) : (
          <FileBrowser
            root={ARTIFACTS_ROOT}
            layout="stacked"
            open={file}
            onOpenChange={setFile}
            onAsk={onAsk}
            placeholder={S.artifactsEmpty}
          />
        )}
      </div>
    </aside>
  );
}
