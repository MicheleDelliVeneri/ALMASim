# ALMASim frontend

SvelteKit 3 (Svelte 5 runes) with TypeScript, Tailwind CSS 4 and Vite 8,
served by `@sveltejs/adapter-node`. It talks to the FastAPI backend in
`../backend`.

## Development

Node 22.17 or newer.

```bash
npm ci
npm run dev        # http://localhost:3000
```

The backend address the browser uses comes from `API_URL`, read on every
request by `src/hooks.server.ts` and injected into the page. Without it the
build-time `VITE_API_URL`, then `http://localhost:8000`, is used
(`src/lib/config.ts`). With Docker, `docker compose up` from the repository
root starts the backend, the database and this dev server together.

## Checks

These are the commands CI runs (`.github/workflows/lint_and_test.yml`):

```bash
npm run gen:api    # regenerate src/lib/api/schema.d.ts from openapi.json
npm run check      # svelte-check
npm run lint       # eslint
npx prettier --check src
npm test           # vitest
npm run build
```

`src/lib/api/openapi.json` is written by `scripts/generate_openapi.py` at the
repository root and guarded by `tests/unit/test_openapi_snapshot.py`.

## Production

```bash
npm run build
API_URL=https://almasim.example.org/api-host node build
```

The image in `Dockerfile` does the same; `docker-compose.prod.yml` runs it.

## Layout

- `src/routes/`: pages (metadata, data, simulations, visualizer, combination,
  imaging)
- `src/lib/api/`: typed API client per backend router, plus the generated
  `schema.d.ts`
- `src/lib/components/`: components by area, and `shared/` for dialogs,
  pickers and status badges
- `src/lib/utils/`: pure TypeScript helpers, each with a Vitest test
