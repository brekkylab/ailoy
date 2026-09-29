import { mount } from "svelte";

import App from "./App.svelte";
import { loadDevEnvKeys } from "./lib/devEnv";
import { applyStoredTheme } from "./lib/theme";
import "./app.css";

// Before the first render, so the window never paints a frame in the other palette.
applyStoredTheme();
void loadDevEnvKeys();

export default mount(App, { target: document.getElementById("app")! });
