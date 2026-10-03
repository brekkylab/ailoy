// What the main panel can be showing.
//
// One list, because two places have to agree on it: the sidebar marks the active row and
// `App` picks the panel. The names are not here — only a conversation is named in the
// title bar, and every other panel writes its own heading.
//
// `settings` has no row in the sidebar: it is reached from the gear in the title bar, which
// is also what marks it active. It is a view all the same, because settings are read and
// compared rather than decided in one stroke — see `SettingsPanel`.

export type MainView = "session" | "workspace" | "artifacts" | "settings";
