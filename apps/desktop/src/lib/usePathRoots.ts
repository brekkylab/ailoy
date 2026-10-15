// The directories a tool call's paths are written against, for the call rows.

import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import * as api from "@/api";
import { pathRoots, type PathRoot } from "@/lib/toolCall";

/** From the workspace the engine reports — the same query the panels read, so a cache hit. */
export function usePathRoots(): PathRoot[] {
  const ws = useQuery({ queryKey: ["workspace"], queryFn: api.workspaceInfo });
  return useMemo(() => pathRoots(ws.data), [ws.data]);
}
