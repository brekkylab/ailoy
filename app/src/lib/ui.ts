// Button looks shared across components — the handful of shadcn variants feat/desktop uses.

const base =
  "inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent text-sm font-medium whitespace-nowrap transition-all outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50";

export const btn = {
  ghostIcon: `${base} size-8 hover:bg-muted hover:text-foreground dark:hover:bg-muted/50`,
  primaryIcon: `${base} size-8 rounded-full bg-primary text-primary-foreground hover:bg-primary/80`,
  primary: `${base} h-8 gap-1.5 px-2.5 bg-primary text-primary-foreground hover:bg-primary/80`,
  outline: `${base} h-7 gap-1 px-2.5 text-[0.8rem] border-border bg-background hover:bg-muted dark:border-input dark:bg-input/30 dark:hover:bg-input/50`,
};

export const input =
  "h-8 w-full min-w-0 rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30";
