// Choosing the model a message is sent to.
//
// A select was the first version, and it stopped fitting once the catalog did: the
// composer can spare a trigger a few words wide, and a list of 250 models across seven
// vendors — Bedrock alone lists Claude, GPT, Llama and Qwen — needs more room to be read
// than a dropdown under a text box has. So the trigger stays small and names the model by
// its vendor's mark and its own name, and choosing opens a dialog: the vendors down the
// left, the chosen vendor's models beside them, and a search across all of them on top.
//
// The keyboard does what it would in a command palette: typing searches, ↑/↓ move through
// the rows on screen, Enter takes the highlighted one, Escape closes without changing
// anything. A model whose vendor has no key is shown and cannot be taken — seeing the
// model you wanted greyed out is the hint that a key is missing, which is the same call
// the settings page's default-model menu makes.

import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { Check, ChevronDown, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import * as api from "@/api";
import { ProviderIcon } from "@/components/icons/ProviderIcon";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { formatPrice, initialVendor, searchModels, vendorsOf, type Vendor } from "@/lib/modelPicker";
import { formatTokens } from "@/lib/tokens";
import { S } from "@/strings";
import type { ModelInfo } from "@/types";

export function ModelPicker({
  value,
  onChange,
  disabled,
}: {
  /** The model id the next message goes to, or `null` before the settings have loaded. */
  value: string | null;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  const models = useQuery({ queryKey: ["models"], queryFn: api.modelsList });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settingsGet });
  const vendors = useMemo(
    () => vendorsOf(settings.data?.providers, models.data),
    [settings.data?.providers, models.data],
  );
  const [open, setOpen] = useState(false);

  const current = models.data?.find((m) => m.id === value) ?? null;
  const currentVendor = current ? vendors.find((v) => v.label === current.provider) : undefined;

  return (
    <>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(true)}
        // The full id on hover: two vendors can carry models with one name, and the trigger
        // only has room for the name.
        title={value ?? undefined}
        aria-label={S.model}
        aria-haspopup="dialog"
        className="flex min-w-0 items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
      >
        {currentVendor && (
          <ProviderIcon providerKey={currentVendor.key} label={currentVendor.label} className="size-3.5 shrink-0" />
        )}
        <span className="truncate">{current?.name ?? value ?? S.model}</span>
        <ChevronDown className="size-3.5 shrink-0" />
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="h-[min(560px,80vh)] grid-rows-[auto_minmax(0,1fr)] gap-0 overflow-hidden p-0 sm:max-w-3xl">
          {/* Mounted per opening, so each one starts from the current model with an empty
              search rather than wherever the last one was left. */}
          {open && (
            <PickerBody
              vendors={vendors}
              value={value}
              onPick={(id) => {
                setOpen(false);
                if (id !== value) onChange(id);
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

function PickerBody({
  vendors,
  value,
  onPick,
}: {
  vendors: Vendor[];
  value: string | null;
  onPick: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [vendorKey, setVendorKey] = useState(() => initialVendor(vendors, value));
  const vendor = vendors.find((v) => v.key === vendorKey) ?? null;
  const searching = query.trim().length > 0;
  const all = useMemo(() => vendors.flatMap((v) => v.models), [vendors]);
  // Memoised because the cursor below is keyed on this array's identity: a search that
  // built a new one each render would put the highlight back at the top on every keystroke
  // and every arrow press.
  const rows = useMemo(
    () => (searching ? searchModels(all, query) : (vendor?.models ?? [])),
    [searching, all, query, vendor],
  );

  // The highlighted row, by position in what is on screen. It follows the list rather than
  // being corrected after it: a new query or another vendor starts at the top — or at the
  // current model, when that is in the list.
  const [cursor, setCursor] = useState({ rows, at: Math.max(0, rows.findIndex((m) => m.id === value)) });
  const at = cursor.rows === rows ? cursor.at : Math.max(0, rows.findIndex((m) => m.id === value));
  const list = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    list.current?.querySelector(`[data-row="${at}"]`)?.scrollIntoView({ block: "nearest" });
  }, [at]);

  const move = (step: number) => {
    if (rows.length === 0) return;
    let i = at;
    // Skip what cannot be taken, but give up after one lap rather than spin on a vendor
    // whose models are all behind a missing key.
    for (let n = 0; n < rows.length; n++) {
      i = (i + step + rows.length) % rows.length;
      if (rows[i].available) break;
    }
    setCursor({ rows, at: i });
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      move(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      move(-1);
    } else if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      e.preventDefault();
      const row = rows[at];
      if (row?.available) onPick(row.id);
    }
  };

  const labelOf = (m: ModelInfo) => vendors.find((v) => v.label === m.provider);

  return (
    <>
      <div className="flex items-center gap-2 border-b px-4 py-3 pr-12">
        <DialogTitle className="sr-only">{S.chooseModel}</DialogTitle>
        <Search className="size-4 shrink-0 text-muted-foreground" />
        <input
          ref={input}
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={S.searchModels}
          aria-label={S.searchModels}
          className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
        />
      </div>
      <div className="grid min-h-0 grid-cols-[12rem_minmax(0,1fr)]">
        <nav aria-label={S.providers} className="min-h-0 space-y-0.5 overflow-y-auto border-r bg-muted/40 p-2">
          {vendors.map((v) => (
            <button
              key={v.key}
              onClick={() => {
                setVendorKey(v.key);
                setQuery("");
                // Back to the box, so the arrows and Enter keep working after a click.
                input.current?.focus();
              }}
              aria-current={!searching && v.key === vendorKey ? "true" : undefined}
              className={cn(
                "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-accent",
                !searching && v.key === vendorKey && "bg-accent font-medium",
                !v.hasKey && "text-muted-foreground",
              )}
            >
              <ProviderIcon providerKey={v.key} label={v.label} className="size-4 shrink-0" />
              <span className="min-w-0 flex-1 truncate">{v.label}</span>
              <span className="text-xs text-muted-foreground tabular-nums">{v.models.length}</span>
            </button>
          ))}
        </nav>
        <div ref={list} role="listbox" aria-label={S.model} className="min-h-0 overflow-y-auto p-2">
          {!searching && vendor && !vendor.hasKey && vendor.models.length > 0 && (
            <p className="mx-2 mb-2 rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">{S.needsKey}</p>
          )}
          {rows.length === 0 && (
            <p className="p-6 text-center text-sm text-muted-foreground">
              {searching ? S.noModelsMatch : S.modelsLoading}
            </p>
          )}
          {rows.map((m, i) => {
            const v = labelOf(m);
            const price = formatPrice(m.cost);
            return (
              <button
                key={m.id}
                data-row={i}
                role="option"
                aria-selected={m.id === value}
                aria-disabled={!m.available}
                disabled={!m.available}
                onClick={() => onPick(m.id)}
                onMouseMove={() => i !== at && m.available && setCursor({ rows, at: i })}
                className={cn(
                  "flex w-full items-center gap-3 rounded-md px-3 py-2 text-left disabled:opacity-45",
                  i === at && m.available && "bg-accent",
                )}
              >
                {/* Across vendors the mark is what tells two `Claude Sonnet 5`s apart. */}
                {searching && v && <ProviderIcon providerKey={v.key} label={v.label} className="size-4 shrink-0" />}
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium">{m.name}</span>
                    {m.reasoning && (
                      <span className="shrink-0 rounded border px-1 py-px text-[10px] text-muted-foreground">
                        {S.reasoning}
                      </span>
                    )}
                  </span>
                  <span className="block truncate font-mono text-[11px] text-muted-foreground">{m.id}</span>
                </span>
                <span className="shrink-0 text-right text-xs text-muted-foreground tabular-nums">
                  {m.context != null && <span className="block">{formatTokens(m.context)} ctx</span>}
                  {price && (
                    <span className="block" title={S.pricePerMtok}>
                      {price}
                    </span>
                  )}
                </span>
                <Check className={cn("size-4 shrink-0", m.id === value ? "text-foreground" : "invisible")} />
              </button>
            );
          })}
        </div>
      </div>
    </>
  );
}
