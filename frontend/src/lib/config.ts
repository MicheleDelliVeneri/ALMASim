import { resolve } from '$app/paths';

/**
 * The API base URL: empty, so requests go to the page's own origin.
 *
 * `almasim serve` serves this app and its API (`/api/v2`) together, so the
 * pages work through one SSH tunnel. In development `vite dev` and
 * `vite preview` forward `/api` to the servers (see `vite.config.ts`). A
 * build-time `VITE_API_URL` still points a build at another origin, which
 * then has to allow it (CORS).
 */
export const API_BASE_URL: string = import.meta.env?.VITE_API_URL ?? '';

/**
 * Where the console is: `VITE_API_URL`, else this page's origin plus the
 * path the console is served under (`almasim serve --base-path`, e.g. Open
 * OnDemand's `/node/<host>/<port>`), which the server puts in the page.
 * Empty outside a browser.
 */
export function consoleUrl(): string {
	if (API_BASE_URL) return API_BASE_URL;
	if (typeof location === 'undefined') return '';
	return location.origin + resolve('/').replace(/\/$/, '');
}
