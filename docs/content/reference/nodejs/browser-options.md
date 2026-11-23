---
title: BrowserOptions
description: Node.js reference for BrowserOptions.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

## Properties

| Property | Type | Description |
| ------ | ------ | ------ |
| <a id="property-isolated"></a> `isolated?` | `boolean` | A browser of the agent's own, apart from the user's: started on first use and closed with the agent. `false` by default: the user's own browser. |
| <a id="property-headless"></a> `headless?` | `boolean` | For `isolated`: no window, as on a server. `false` by default. |
| <a id="property-executable"></a> `executable?` | `string` | For `isolated`: the Chromium-based browser to start, by its executable; else one installed here (Chrome, Chromium, Edge or Brave). |
| <a id="property-profile"></a> `profile?` | `string` | For `isolated`: the user data directory it keeps, so sign-ins last; a fresh one by default. |
| <a id="property-userbrowser"></a> `userBrowser?` | \| `"chrome"` \| `"chrome-beta"` \| `"chrome-dev"` \| `"chrome-canary"` \| `"chromium"` \| `"edge"` \| `"brave"` \| \{ `userDataDir`: `string`; \} | For the user's browser, which one: `'chrome'`, `'chrome-beta'`, `'chrome-dev'`, `'chrome-canary'`, `'chromium'`, `'edge'`, `'brave'`, or any other by `{ userDataDir }`; with none, whichever turned remote debugging on last. |
| <a id="property-careful"></a> `careful?` | `boolean` | Whether the agent is told to use sites as a person would, so sites checking for bots are less likely to turn it away. `true` by default. |
| <a id="property-starturl"></a> `startUrl?` | `string` | The page the agent's tab opens on. |
