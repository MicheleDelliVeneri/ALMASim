# Build stage: the web app as static files (adapter-static, fallback 200.html)
FROM node:22-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage: `vite preview` serves the build and forwards /api to the
# console, as configured in vite.config.ts: /api/v2 to ALMASIM_API_URL with
# ALMASIM_TOKEN. This is the compose set-up for a laptop; on a cluster,
# `almasim serve` serves the same build itself (it is packaged in the wheel)
# and no frontend container is needed.
FROM builder AS production

ENV ALMASIM_API_URL=http://console:8765

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (r) => {if(r.statusCode !== 200) throw new Error(r.statusCode)})"

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "3000"]
