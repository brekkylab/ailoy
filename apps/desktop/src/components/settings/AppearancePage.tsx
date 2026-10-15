// The appearance page: what the window looks like.
//
// Nothing here goes through the engine. A palette is a preference of this window on this
// machine, wanted before the first IPC call resolves — see `lib/theme`, which is also where
// the attribute the whole stylesheet is written around gets set.

import { cn } from "cn";
import { Monitor, Moon, Sun } from "lucide-react";
import { useState } from "react";

import { Section } from "@/components/settings/fields";
import { readTheme, setTheme, type Theme } from "@/lib/theme";
import { S } from "@/strings";

const OPTIONS: { id: Theme; label: string; Icon: typeof Monitor }[] = [
  { id: "system", label: S.themeSystem, Icon: Monitor },
  { id: "light", label: S.themeLight, Icon: Sun },
  { id: "dark", label: S.themeDark, Icon: Moon },
];

export function AppearancePage() {
  // Seeded from storage rather than from a prop: the attribute is already set — `main.tsx`
  // applies it before the first render — and this is the component that changes it, so
  // there is nothing above to hold it.
  const [theme, pick] = useState<Theme>(readTheme);
  const choose = (next: Theme) => {
    pick(next);
    setTheme(next);
  };

  return (
    <div className="space-y-8">
      <Section title={S.appearanceSection}>
        <div className="space-y-2">
          {/* Three buttons and not a menu: the choice is small, and each option can carry
              the mark that says what it means faster than its name does. */}
          <div
            role="radiogroup"
            aria-label={S.theme}
            className="flex gap-1 rounded-lg border bg-muted/20 p-1"
          >
            {OPTIONS.map(({ id, label, Icon }) => (
              <button
                key={id}
                role="radio"
                aria-checked={theme === id}
                onClick={() => choose(id)}
                className={cn(
                  "flex flex-1 items-center justify-center gap-2 rounded-md border border-transparent px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                  theme === id && "border-primary/30 bg-primary/10 font-medium text-foreground",
                )}
              >
                <Icon className="size-4 shrink-0" />
                {label}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">{S.themeHint}</p>
        </div>
      </Section>
    </div>
  );
}
