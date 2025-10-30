import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "ailoy-node",
          root: "./tests",
          environment: "node",
        },
      },
    ],
  },
});
