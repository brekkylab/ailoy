// Settings, in the main panel, as pages.
//
// A panel and not a dialog. Keys are the one thing a new install cannot start without, and
// a provider's pane is where the user finds out whether the key they pasted took — which is
// reading, comparing and coming back, not the one decision a modal is for. A modal also
// takes the window hostage: the no-key banner that sent the user here is behind it, and so
// is the model picker the key is about to fill.
//
// Pages, because settings only grow. A strip across the top and one component per page: a
// page is a row in `PAGES` and a file in `components/settings`, and it brings its own
// queries with it — nothing here is shared state to thread through, and a page the user has
// not opened has not mounted, so it costs nothing to add one.

import { Cpu, Palette } from "lucide-react";

import { AppearancePage } from "@/components/settings/AppearancePage";
import { ModelsPage } from "@/components/settings/ModelsPage";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { S } from "@/strings";

const PAGES = [
  { id: "models", label: S.modelsPage, Icon: Cpu, Page: ModelsPage },
  { id: "appearance", label: S.appearancePage, Icon: Palette, Page: AppearancePage },
];

export function SettingsPanel() {
  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {/* Uncontrolled: which page the user was last on is not worth remembering across a
          relaunch, and the panel is rebuilt each time it opens — which lands on the models
          page, the one a new install has to visit. */}
      <Tabs defaultValue={PAGES[0].id} className="flex min-h-0 flex-1 flex-col">
        {/* The page strip is all this bar holds: the panel's name is the title bar's. */}
        <div className="border-b px-6 pb-2">
          <TabsList aria-label={S.settings} className="gap-1">
            {PAGES.map(({ id, label, Icon }) => (
              <TabsTab key={id} value={id}>
                <Icon className="size-4 shrink-0" />
                {label}
              </TabsTab>
            ))}
          </TabsList>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-6 pt-5 pb-10">
          <div className="mx-auto max-w-3xl">
            {PAGES.map(({ id, Page }) => (
              <TabsPanel key={id} value={id}>
                <Page />
              </TabsPanel>
            ))}
          </div>
        </div>
      </Tabs>
    </section>
  );
}
