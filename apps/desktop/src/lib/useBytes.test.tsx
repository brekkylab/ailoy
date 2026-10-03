// @vitest-environment jsdom
//
// The one test in this app that renders anything, and it is here because the bug it guards
// against cannot be seen anywhere else: a hook that returns a fresh object on every render
// type-checks, passes every unit test, and takes the file viewers apart. The value below is
// what five viewers key an effect on, and opening a document sets state — so an identity that
// changes per render opens and destroys the document forever, which reaches the window as a
// blank pane and nothing in the log.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useBytes } from "./useBytes";

const reads = vi.hoisted(() => ({ calls: 0 }));
vi.mock("@/api", () => ({
  fsReadBytes: () => {
    reads.calls += 1;
    return Promise.resolve(new ArrayBuffer(8));
  },
}));
vi.mock("@/lib/report", () => ({ report: () => {} }));

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

afterEach(() => {
  reads.calls = 0;
});

describe("useBytes", () => {
  it("returns the same value while the bytes have not changed", async () => {
    const { result, rerender } = renderHook(() => useBytes("/a.pdf"), { wrapper });
    await waitFor(() => expect(result.current.state).toBe("ready"));

    const first = result.current;
    rerender();
    rerender();
    // Not `toEqual`: it is the identity that matters, because that is what an effect's
    // dependency array compares.
    expect(result.current).toBe(first);
  });

  it("reads a file once, however many times a viewer renders", async () => {
    const { result, rerender } = renderHook(() => useBytes("/b.pdf"), { wrapper });
    await waitFor(() => expect(result.current.state).toBe("ready"));
    rerender();
    expect(reads.calls).toBe(1);
  });
});
