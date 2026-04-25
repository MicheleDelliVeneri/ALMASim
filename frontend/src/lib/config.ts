/**
 * Runtime-resolvable API base URL.
 *
 * Resolution order (first non-empty wins):
 *   1. Browser:  window.__ALMASIM_CONFIG__.apiUrl
 *      (injected at request time by `hooks.server.ts` from $env API_URL)
 *   2. Server:   process.env.API_URL
 *   3. Build-time fallback: import.meta.env.VITE_API_URL
 *   4. Hard fallback: http://localhost:8000
 *
 * This lets the same prebuilt Docker image talk to any backend by setting
 * the API_URL environment variable at container runtime, without rebuilding.
 */

declare global {
	interface Window {
		__ALMASIM_CONFIG__?: { apiUrl?: string };
	}
}

const BUILD_TIME_FALLBACK =
	(typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || 'http://localhost:8000';

function resolveApiBaseUrl(): string {
	if (typeof window !== 'undefined') {
		const fromWindow = window.__ALMASIM_CONFIG__?.apiUrl;
		if (fromWindow) return fromWindow;
	}
	if (typeof process !== 'undefined' && process.env?.API_URL) {
		return process.env.API_URL;
	}
	return BUILD_TIME_FALLBACK;
}

export const API_BASE_URL: string = resolveApiBaseUrl();

export function getApiBaseUrl(): string {
	return resolveApiBaseUrl();
}
