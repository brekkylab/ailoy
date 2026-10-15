"use client"

import { Tabs as TabsPrimitive } from "@base-ui/react/tabs"
import { cn } from "cn"

const Tabs = TabsPrimitive.Root

function TabsList({ className, ...props }: TabsPrimitive.List.Props) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn("flex gap-1 data-[orientation=vertical]:flex-col", className)}
      {...props}
    />
  )
}

// Selected state read off `aria-selected`, which the role guarantees, rather than off a
// state attribute: Base UI derives those from its own state object and the name has moved
// between versions.
//
// It is marked in the primary tint and not with `bg-accent`, which is what the sidebar
// marks a row with: `--accent` is a 4.5% wash in this palette, and a tab rail is a control
// whose whole job is to say which of several things you are looking at. The border is on
// every tab, transparent until selected, so selecting one does not move the others; and it
// is a border rather than a ring so that a focus ring still has somewhere to draw.
function TabsTab({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-tab"
      className={cn(
        "flex items-center gap-2 rounded-md border border-transparent px-2.5 py-1.5 text-sm whitespace-nowrap text-muted-foreground transition-colors select-none hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 aria-selected:border-primary/30 aria-selected:bg-primary/10 aria-selected:font-medium aria-selected:text-foreground",
        className
      )}
      {...props}
    />
  )
}

function TabsPanel({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-panel"
      className={cn("min-w-0 outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTab, TabsPanel }
