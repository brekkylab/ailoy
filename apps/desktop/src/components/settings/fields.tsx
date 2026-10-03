// The rows a settings page is built from.
//
// Shared because a setting should look like a setting wherever it is: a heading over a
// group, a menu over a closed set, a number that commits when you leave it. A page adds
// its own markup only where it has something these three cannot say.

import { useId, useState } from "react";

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { RegionRouting } from "@/types";

/** A group of settings under a heading. */
export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">{title}</h2>
      {children}
    </section>
  );
}

/** One closed set, as a labelled menu that saves the moment it is picked. */
export function Choice({
  label,
  hint,
  value,
  options,
  disabled,
  onPick,
}: {
  label: string;
  hint: string;
  /** The stored choice, already defaulted by the engine. */
  value: string | null;
  options: RegionRouting[];
  disabled: boolean;
  onPick: (id: string) => void;
}) {
  const id = useId();
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Select
        value={value}
        disabled={disabled}
        onValueChange={(v) => {
          if (typeof v === "string" && v && v !== value) onPick(v);
        }}
      >
        <SelectTrigger id={id} className="w-full" aria-label={label}>
          {/* Same reason as the model picker: Base UI reads the trigger's text from the
              selected item, which lives in a portal that has not mounted until the list is
              opened. */}
          <SelectValue>
            {(picked: unknown) =>
              options.find((o) => o.id === picked)?.label ?? (typeof picked === "string" ? picked : label)
            }
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.id} value={o.id}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

/**
 * A number that saves when the field loses focus. The engine happily accepts `0` for the
 * limits this draws, which would end every run before it produced anything, so the floor
 * is 1 here; anything else — blank, a decimal, unchanged — snaps back to the stored value
 * rather than sending a patch.
 *
 * The draft starts from `value` and is re-seeded by remounting (the caller keys this on
 * the stored number), which is why there is no effect syncing prop into state.
 */
export function NumberSetting({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: number;
  onCommit: (n: number) => void;
}) {
  const id = useId();
  const [text, setText] = useState(() => String(value));

  const commit = () => {
    const n = Number(text);
    if (Number.isInteger(n) && n >= 1 && n !== value) onCommit(n);
    else setText(String(value));
  };

  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type="number"
        min={1}
        step={1}
        inputMode="numeric"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
      />
    </div>
  );
}
