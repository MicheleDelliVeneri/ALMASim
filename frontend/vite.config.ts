import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type ProxyOptions } from 'vite';

/**
 * Where `vite dev` and `vite preview` send `/api/v2`: the console server
 * (`almasim serve`). ALMASIM_API_URL names it (a URL, or the path of its
 * socket) and ALMASIM_TOKEN its token; without them, the server recorded in
 * ~/.almasim/serve.json is used, so a running `almasim serve` just works.
 */
function consoleProxy(): ProxyOptions {
	let recorded: { token?: string; socket?: string | null; port?: number | null } = {};
	try {
		recorded = JSON.parse(readFileSync(join(homedir(), '.almasim', 'serve.json'), 'utf-8'));
	} catch {
		// no server recorded
	}
	const where =
		process.env.ALMASIM_API_URL ?? recorded.socket ?? `http://127.0.0.1:${recorded.port ?? 8765}`;
	const token = process.env.ALMASIM_TOKEN ?? recorded.token;
	return {
		// A socket path goes in `target.socketPath`; the host is then a placeholder.
		target: where.startsWith('/') ? { socketPath: where, host: 'localhost' } : where,
		changeOrigin: true,
		headers: token ? { Authorization: `Bearer ${token}` } : undefined
	};
}

const proxy = { '/api/v2': consoleProxy() };

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			preprocess: vitePreprocess(),
			// One page that routes in the browser; `almasim serve` returns it for
			// any path that is not a file (src/almasim/web/app.py).
			adapter: adapter({ fallback: '200.html' })
		})
	],
	server: { port: 3000, strictPort: true, proxy },
	preview: { port: 3000, strictPort: true, proxy }
});
