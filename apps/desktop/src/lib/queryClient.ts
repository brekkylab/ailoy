// The one query client, and what it takes an answer to be worth.
//
// Its own module rather than a few lines in `App`, because the policy below *is* the caching
// strategy for the file tree: anything that builds a client without it — a second entry point,
// a preview page — is a window that asks the engine for everything twice.

import { QueryClient } from "@tanstack/react-query";

import { BYTES_GC_MS, BYTES_KEY } from "@/lib/bytes";
import { ARTIFACTS_ROOT } from "@/paths";

/**
 * How long a listing or a file answer is taken as still true.
 *
 * Nothing in the window writes to the workspace, so what is in the tree changes for two
 * reasons and both invalidate these keys where they happen: a run finishes (`UsageBar`) or a
 * source is connected or removed (`SourceDialog`, `MountDialogs`). Between those, re-expanding
 * a directory or re-opening a file is the same answer — and on Notion or S3 it is a round trip
 * to hear it, measured at 0.4-2.5s for a page. Refresh in the workspace panel asks anyway,
 * which is what makes this a cache rather than a guess about how often files change.
 */
export const FS_STALE_MS = 30_000;

/** How long an answer nothing is looking at is kept, for the walk back up a tree. */
export const FS_GC_MS = 30 * 60_000;

export function makeQueryClient(): QueryClient {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
  });
  qc.setQueryDefaults(["fs"], { staleTime: FS_STALE_MS, gcTime: FS_GC_MS });
  qc.setQueryDefaults(["file"], { staleTime: FS_STALE_MS, gcTime: FS_GC_MS });
  // The bytes have their own staleness — never, until something invalidates them — set where
  // they are read, and their own bound, set where they are trimmed.
  qc.setQueryDefaults([BYTES_KEY], { gcTime: BYTES_GC_MS });
  return qc;
}

/**
 * Whether a finished run could have changed what this query holds.
 *
 * A run writes to one place. The agent's console mounts the workspace as *context*, which
 * cortex refuses writes under, and its output goes to the artifacts tree — so after a run
 * the listings and files under `/artifacts` are the ones worth asking about again, and
 * nothing else in the tree is.
 *
 * That distinction is not housekeeping. The rows of a Notion tree are `["file", …]` queries,
 * one per visible page, and each is a page render on the far side: invalidating them all at
 * the end of every run re-read the whole visible tree, over the network, for a source the
 * agent cannot write to at all.
 */
export function changedByRun(key: readonly unknown[]): boolean {
  const [kind, path] = key;
  if (kind !== "fs" && kind !== "file" && kind !== BYTES_KEY) return false;
  return (
    typeof path === "string" &&
    (path === ARTIFACTS_ROOT || path.startsWith(`${ARTIFACTS_ROOT}/`))
  );
}
