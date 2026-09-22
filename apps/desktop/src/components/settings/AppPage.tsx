// The app page: what this install is, and where to look when it misbehaves.

import { useMutation, useQuery } from "@tanstack/react-query";

import * as api from "@/api";
import { Section } from "@/components/settings/fields";
import { Button } from "@/components/ui/button";
import { S } from "@/strings";

export function AppPage() {
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  // A mutation and not a bare call, so a failure to open the folder has somewhere to go.
  const openLogs = useMutation({ mutationFn: api.openLogs });

  // Two sections rather than one under the page's own name, which the tab already says.
  return (
    <div className="space-y-8">
      <Section title={S.workspace}>
        {ws.data ? (
          <p className="text-xs text-muted-foreground">
            <span className="font-mono">{ws.data.mountpoint}</span> ·{" "}
            {ws.data.status.status === "mounted"
              ? S.mounted
              : `${S.degradedShort} (${ws.data.status.reason})`}
          </p>
        ) : (
          ws.isError && <p className="text-sm text-destructive">{api.messageOf(ws.error)}</p>
        )}
      </Section>

      <Section title={S.logsSection}>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">{S.logsHint}</p>
          <Button
            size="sm"
            variant="link"
            className="h-auto px-0"
            onClick={() => openLogs.mutate()}
            disabled={openLogs.isPending}
          >
            {S.openLogs}
          </Button>
          {openLogs.isError && <p className="text-sm text-destructive">{api.messageOf(openLogs.error)}</p>}
        </div>
      </Section>
    </div>
  );
}
