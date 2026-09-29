// A file as the viewers see it: what the registry picks a viewer by, and what viewers
// name, size and resolve against.

export interface Entry {
  name: string;
  /** Path in the tree it was opened from, one name per segment. */
  segments: string[];
  type: "file" | "folder";
  size: number | null;
  /** Milliseconds since the epoch, when known. */
  modified: number | null;
}
