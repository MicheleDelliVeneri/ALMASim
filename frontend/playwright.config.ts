import { defineConfig } from '@playwright/test';

/**
 * End-to-end tests (web console plan, U3.9): the built app, served by a real
 * `almasim serve` whose commands are instant stand-ins (e2e/serve.sh,
 * e2e/fake_almasim.py). Build the app first (`npm run build`).
 *
 * E2E_BASE_PATH serves the console under a URL prefix, as Open OnDemand's
 * proxy does (`/node/<host>/<port>`): the same tests then run under it.
 */
const PORT = Number(process.env.E2E_PORT ?? 8765);
export const TOKEN = process.env.ALMASIM_TOKEN ?? 'e2e-console-token-0123456789';
export const E2E_ROOT = process.env.E2E_ROOT ?? '/tmp/almasim-e2e';
export const BASE_PATH = process.env.E2E_BASE_PATH ?? '';

export default defineConfig({
	testDir: 'e2e',
	timeout: 90_000,
	expect: { timeout: 30_000 },
	workers: 1, // one console, one runs folder
	retries: process.env.CI ? 1 : 0,
	reporter: process.env.CI ? [['github'], ['list']] : 'list',
	use: {
		baseURL: `http://127.0.0.1:${PORT}${BASE_PATH}/`,
		extraHTTPHeaders: { Authorization: `Bearer ${TOKEN}` },
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	webServer: {
		command: 'bash e2e/serve.sh',
		// Without the token the health check answers 401, which says it is up.
		url: `http://127.0.0.1:${PORT}${BASE_PATH}/api/v2/health`,
		reuseExistingServer: false,
		timeout: 120_000,
		env: { ALMASIM_TOKEN: TOKEN, E2E_ROOT, E2E_PORT: String(PORT), E2E_BASE_PATH: BASE_PATH }
	}
});
