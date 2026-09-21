// What the main panel can be showing.
//
// One list, because two places have to agree on it: the sidebar marks the active row and
// `App` picks the panel. The names are not here — only a conversation is named in the
// title bar, and the other two panels write their own heading.

export type MainView = "session" | "workspace" | "artifacts";
