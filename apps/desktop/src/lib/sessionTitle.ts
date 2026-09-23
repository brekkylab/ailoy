import type { SessionSummary } from "@/types";

/**
 * What the title bar names, for the session the window is showing.
 *
 * `null` means a blank bar. Three different situations end there and they all look the
 * same from the window: no session is open, the list has not arrived yet, and the chosen
 * session is no longer in the list. The last two are the reason this returns `null`
 * rather than falling back to anything — the id outlives the list across a delete and a
 * refetch, so a bar that guessed would name the wrong conversation for as long as the
 * list took to catch up, which is exactly when the user is looking at it.
 *
 * An empty stored title is blank too. A heading with nothing in it is still a heading.
 */
export function sessionTitle(
  sessionId: string | null,
  sessions: SessionSummary[] | undefined,
): string | null {
  if (!sessionId || !sessions) return null;
  return sessions.find((s) => s.id === sessionId)?.title || null;
}
