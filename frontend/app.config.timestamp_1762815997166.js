// app.config.ts
import { defineConfig } from "@solidjs/start/config";
var app_config_default = defineConfig({
  appRoot: "./app",
  server: {
    preset: "node-server"
  },
  vite: {
    server: {
      port: 3e3,
      strictPort: true,
      proxy: {
        "/api": {
          target: "http://localhost:8000",
          changeOrigin: true
        }
      }
    }
  }
});
export {
  app_config_default as default
};
