/**
 * The console API (`/api/v2`, served by `almasim serve`), typed from its
 * OpenAPI schema: `schema.d.ts` is generated from `openapi.json` by
 * `npm run gen:api`.
 *
 *   const { data } = await api.GET('/api/v2/jobs', { params: { query: { owner: 'all' } } });
 *
 * A failed request throws an `ApiRequestError` (`../client.ts`), so
 * `data` is always there when the call returns.
 */
import createClient, { type Middleware } from 'openapi-fetch';
import { API_BASE_URL } from '#lib/config.js';
import { ApiRequestError, formatErrorDetail } from '../client.js';
import type { components, paths } from './schema.js';

export type Run = components['schemas']['Run'];
export type RunDetail = components['schemas']['RunDetail'];
export type JobState = components['schemas']['JobState'];
export type Listing = components['schemas']['Listing'];
export type Entry = components['schemas']['Entry'];
export type Root = components['schemas']['Root'];

export const throwOnError: Middleware = {
	async onResponse({ response }) {
		if (response.ok) return undefined;
		const body = await response
			.clone()
			.json()
			.catch(() => ({ detail: `HTTP ${response.status}: ${response.statusText}` }));
		const message = formatErrorDetail(body?.detail) || `HTTP ${response.status}`;
		throw new ApiRequestError(message, response.status, body);
	}
};

export function createApi(
	// The page's own origin, spelled out: a Request outside a browser page
	// (tests, workers) cannot resolve a bare path.
	baseUrl: string = API_BASE_URL || (typeof location === 'undefined' ? '' : location.origin),
	// Looked up on each call, not when the client is made (tests stub it).
	fetchImpl: (request: Request) => Promise<Response> = (request) => globalThis.fetch(request)
) {
	const client = createClient<paths>({ baseUrl, credentials: 'same-origin', fetch: fetchImpl });
	client.use(throwOnError);
	return client;
}

export const api = createApi();
