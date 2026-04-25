import type { Handle } from '@sveltejs/kit';

/**
 * Inject runtime configuration into the HTML response so the browser can
 * read it via `window.__ALMASIM_CONFIG__`. The values come from environment
 * variables read at request time, which means the same prebuilt image can
 * be deployed against different backends just by changing API_URL.
 *
 * The placeholder `%almasim.runtimeConfig%` lives in `src/app.html`.
 */
const PLACEHOLDER = '%almasim.runtimeConfig%';

export const handle: Handle = async ({ event, resolve }) => {
	const apiUrl = process.env.API_URL ?? '';
	const configJson = JSON.stringify({ apiUrl });
	// Escape `<` so the JSON can't break out of the <script> block.
	const safeJson = configJson.replace(/</g, '\\u003c');
	const snippet = `<script>window.__ALMASIM_CONFIG__=${safeJson};</script>`;

	return resolve(event, {
		transformPageChunk: ({ html }) => html.replace(PLACEHOLDER, snippet)
	});
};
