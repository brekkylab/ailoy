import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.tsx";
import { applyStoredTheme } from "./lib/theme.ts";
import "./index.css";

// Before the first render: a palette applied from inside a component is one frame of the
// other one first. `App`'s `useSystemTheme` keeps Tailwind's `.dark` class in step from here.
applyStoredTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
