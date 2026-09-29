# Ailoy app

A Tauri 2 desktop app with a Svelte 5 + TypeScript frontend (Vite, Tailwind v4, brekkylab's
`@brekkylab/ui` tokens). The UI follows `feat/desktop`'s `apps/desktop`.

This is UI only for now: nothing talks to ailoy. Sessions, messages, and API keys live in
memory (`src/lib/store.svelte.ts`, seeded from `src/lib/mock.ts`), and a sent message gets
a placeholder reply.

## Run

```sh
npm install          # needs access to the private brekkylab/ui repo
npm run tauri dev    # the desktop app
npm run dev          # or just the frontend, in a browser at http://localhost:1420
```

`npm run check` type-checks; `npm run tauri build` bundles.

### API keys in dev

A debug build reads the nearest `.env` at startup — `app/.env`, else the repo root's — and
fills Settings with its keys. Variables set in the shell win over the
file. The names are the ones ailoy reads; see `.env.example`:

| Provider | Variable |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK`, region from `AWS_REGION` / `AWS_DEFAULT_REGION` |

Release builds never read a `.env`. `npm run dev` in a plain browser has no Rust side, so
it starts without keys.

## Layout

- `src/App.svelte` — the shell: title bar, sidebar, and the main panel for the current view
- `src/components/` — `TitleBar`, `Sidebar`, `Thread` (+ `thread/`), `SettingsPanel` (+ `settings/`),
  and placeholder `ContextPanel` / `AgentPanel` / `ArtifactsPanel`
- `src/lib/` — theme, session grouping, the in-memory store, and mock data
- `src-tauri/` — the Rust shell, its own Cargo workspace so the root `cargo test` doesn't build it
