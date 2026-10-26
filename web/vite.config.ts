/// <reference types="vitest/config" />
// The build goes into the Python package, so `cattle3d serve` and `cattle3d demo`
// serve the console without Node on the farm computer. `npm run dev` proxies the
// API to a server started with `cattle3d serve` or `cattle3d demo` on port 8000.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/cattle3d/web/static", emptyOutDir: true, chunkSizeWarningLimit: 400 },
  server: { proxy: { "/v1": api, "/health": api, "/figures": api } },
  test: { environment: "jsdom" },
});
