// The strip along the top of the window, in two segments.
//
// The window runs with `titleBarStyle: "Overlay"`, so macOS draws the traffic lights
// *over* the webview and gives us no title bar of its own. That buys the whole window
// height for the app, and costs us two things we now have to supply: a left inset the
// lights can sit in without anything underneath them, and a drag region, because an
// overlay title bar is not draggable by itself.
//
// `data-tauri-drag-region="deep"` makes the whole bar draggable. Tauri walks up from
// whatever was clicked, and stops at the first clickable element it meets — a `button`
// among them — so the two buttons below are excluded from the drag region without being
// marked, and everything else in the bar drags. Dragging also needs
// `core:window:allow-start-dragging` in the capability file, and the double-click that
// zooms the window needs `core:window:allow-internal-toggle-maximize`; neither is in
// `core:default`, whose window set is empty, so without them the bar looks right and
// simply does not move.
//
// The left segment is the sidebar's. It holds the inset and the collapse toggle, and it
// keeps the sidebar's width so the divider below it runs straight. When the sidebar is
// hidden the segment shrinks to just the inset and the toggle, which is why the toggle
// sits next to the lights rather than at the segment's far edge — it has to stay in the
// same place in both states, or it moves out from under the cursor that just clicked it.
//
// The right segment is the session's: its title, and the way into the app's settings.
// Settings used to live in the sidebar, where hiding the sidebar would have taken it with
// it. The gear opens a panel rather than a dialog, so it is a place the window can *be* —
// hence `settingsActive`, which marks it the way the sidebar marks its own rows.

import { cn } from "cn";
import { PanelLeftClose, PanelLeftOpen, Settings as SettingsIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { S } from "@/strings";

/** The bar's height. The traffic lights are centred against this, see below. */
const BAR_H = "h-11"; // 44px

/**
 * The room the macOS traffic lights need at the left. Nothing interactive may sit inside
 * it: the system draws the lights on top of whatever we render there.
 *
 * This pairs with `trafficLightPosition` in `tauri.conf.json`, and the two have to be read
 * together. `x` is the gap to the left of the first button, so 16 puts the three 12pt
 * buttons, 20pt apart, from x=16 to x=68, and the 78px here is that span plus room before
 * the toggle.
 *
 * `y` is *not* the matching gap above them, though it moves them point for point. The
 * macOS backend resizes the title bar container to `buttonHeight + y` and pins it to the
 * top of the window, and the buttons hold their own offset inside it, so the gap you see
 * is `y` minus a few points. The 20 here was landed by eye against this 44px bar, where
 * 16 sat too high and 24 too low; it is not `(44 - 12) / 2`, and deriving it that way is
 * what put the lights in the wrong place twice.
 *
 * So treat it as calibrated, not computed. The slope is 1: if the lights are N points off,
 * move `y` by N. Change the bar's height and they need re-centring by the same rule.
 *
 * Read the result before trusting a change. This applies when the window is built, so a
 * running `tauri dev` has to relaunch the app, not just rebuild it — until it does, the
 * old spacing is still on screen and looks like the setting was ignored, which is its own
 * way of sending you after the wrong bug.
 */
const TRAFFIC_LIGHTS = "pl-[78px]";

export function TitleBar({
  title,
  collapsed,
  onToggleSidebar,
  settingsActive,
  onOpenSettings,
}: {
  /** The current session's title, or `null` when no session is open. */
  title: string | null;
  collapsed: boolean;
  onToggleSidebar: () => void;
  /** Whether the settings panel is what the main panel is showing. */
  settingsActive: boolean;
  onOpenSettings: () => void;
}) {
  return (
    <header
      data-tauri-drag-region="deep"
      // No rule under the bar: the thread fades out beneath it instead, and a line plus a
      // fade reads as two separate edges.
      className={cn("col-span-full flex shrink-0 items-stretch", BAR_H)}
    >
      <div className={cn("flex shrink-0 items-center", TRAFFIC_LIGHTS, !collapsed && "w-[var(--sidebar-w)]")}>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSidebar}
          aria-label={collapsed ? S.expandSidebar : S.collapseSidebar}
          aria-expanded={!collapsed}
        >
          {collapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
        </Button>
      </div>
      <div className={cn("flex min-w-0 flex-1 items-center gap-2 pr-2 pl-3", !collapsed && "border-l")}>
        {/* Empty until a session is open. A heading with nothing in it is still a heading,
            so it is only rendered when it has something to name. */}
        {title && <h1 className="truncate text-sm font-medium">{title}</h1>}
        <Button
          variant="ghost"
          size="icon"
          className={cn("ml-auto", settingsActive && "bg-accent text-accent-foreground")}
          onClick={onOpenSettings}
          aria-label={S.settings}
          aria-pressed={settingsActive}
        >
          <SettingsIcon className="size-4" />
        </Button>
      </div>
    </header>
  );
}
