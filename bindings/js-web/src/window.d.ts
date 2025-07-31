import type { MainModule } from "./ailoy_js_web";

declare global {
  interface Window {
    getModule: () => Promise<MainModule>;
  }
}
