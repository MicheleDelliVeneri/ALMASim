# ALMASim frontend

The ALMASim web console: SvelteKit 3 (Svelte 5 runes) with TypeScript,
Tailwind CSS 4 and Vite 8, built by `@sveltejs/adapter-static` into one page
that routes in the browser (`build/200.html`). It calls `/api` on its own
origin: API v2 is the console (`almasim serve`, `src/almasim/web`), and API
v1 the older backend in `../backend`, which its remaining pages still use.

## Development

Node 22.17 or newer.

```bash
npm ci
npm run dev        # http://localhost:3000
```

`vite dev` (and `vite preview`) forward `/api` (`vite.config.ts`):

- `/api/v2` to the console: the one `almasim serve` recorded in
  `~/.almasim/serve.json` (its socket or port, and its token), or
  `ALMASIM_API_URL` (a URL or a socket path) with `ALMASIM_TOKEN`;
- `/api/v1` to `LEGACY_API_URL`, by default `http://localhost:8000`.

So start `almasim serve` (and the old backend, for the old pages), then
`npm run dev`. With Docker, `docker compose up` from the repository root
starts the backend, the console, the database and this dev server together.

## Checks

These are the commands CI runs (`.github/workflows/lint_and_test.yml`):

```bash
npm run gen:api    # regenerate both schema.d.ts from their openapi.json
npm run check      # svelte-check
npm run lint       # eslint
npx prettier --check src
npm test           # vitest: unit tests in Node, components in jsdom
npm run build
```

End-to-end tests (`e2e/`, CI's `e2e` job) drive the built app in Chromium
against a real `almasim serve`, its commands replaced by instant stand-ins
(`e2e/fake_almasim.py`, through `ALMASIM_JOB_COMMAND`), and check every page
with axe:

```bash
npm run build
ALMASIM=../.venv/bin/almasim PYTHON=../.venv/bin/python npm run test:e2e
```

`src/lib/api/openapi.json` (v1) and `src/lib/api/v2/openapi.json` (v2) are
written by `scripts/generate_openapi.py` at the repository root and guarded
by `tests/unit/test_openapi_snapshot.py`.

## Production

`npm run build` writes the app to `build/`. The Python distributions carry
it (`hatch_build.py`): the wheel installs it as `almasim/web/static`, and
`almasim serve` serves it with the API, so a cluster needs no Node at all.
From a source checkout, `almasim serve` serves `frontend/build` directly;
`ALMASIM_WEB_DIR` points it anywhere else.

The image in `Dockerfile` serves the build with `vite preview` and the same
forwarding, for `docker-compose.prod.yml` on a laptop.

## Layout

- `src/routes/`: pages (metadata, data, simulations, visualizer, combination,
  imaging)
- `src/lib/api/`: typed API clients: v1 per backend router with the generated
  `schema.d.ts`, and `v2/` for the console (openapi-fetch)
- `src/lib/stores/`: the live jobs store, fed by `/api/v2/events`
- `src/lib/components/`: components by area, and `shared/` for the dialog,
  pickers, toasts and status badges
- `src/lib/utils/`: pure TypeScript helpers, each with a Vitest test
