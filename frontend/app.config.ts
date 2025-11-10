import { defineConfig } from "@solidjs/start/config";

export default defineConfig({
  appRoot: "./app",
  server: {
    preset: "node-server",
  },
  vite: {
    server: {
      port: 3000,
      strictPort: true,
      proxy: {
        "/api": {
          target: "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  },
});
