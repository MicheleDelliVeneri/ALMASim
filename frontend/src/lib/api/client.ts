/** API client for backend communication */
import { API_BASE_URL } from '#lib/config.js';

/** FastAPI error bodies: `detail` is a string, or a list of validation errors (422). */
export interface ApiError {
	detail?: unknown;
}

/** An HTTP error from the backend, with a readable message and the raw response body. */
export class ApiRequestError extends Error {
	constructor(
		message: string,
		readonly status: number,
		readonly body: unknown
	) {
		super(message);
		this.name = 'ApiRequestError';
	}
}

/** Turn a FastAPI `detail` into text: validation errors become "field: message; ...". */
export function formatErrorDetail(detail: unknown): string {
	if (typeof detail === 'string') return detail;
	if (Array.isArray(detail)) {
		return detail
			.map((item) => {
				if (item && typeof item === 'object' && 'msg' in item) {
					const loc = Array.isArray((item as { loc?: unknown }).loc)
						? (item as { loc: unknown[] }).loc.filter((part) => part !== 'body').join('.')
						: '';
					return loc
						? `${loc}: ${String((item as { msg: unknown }).msg)}`
						: String((item as { msg: unknown }).msg);
				}
				return typeof item === 'string' ? item : JSON.stringify(item);
			})
			.join('; ');
	}
	if (detail && typeof detail === 'object') return JSON.stringify(detail);
	return '';
}

/** The request took longer than its `timeoutMs`. */
export class ApiTimeoutError extends Error {
	constructor(readonly timeoutMs: number) {
		super(`Timed out after ${Math.round(timeoutMs / 1000)} s`);
		this.name = 'ApiTimeoutError';
	}
}

/** Per-request options: a caller's abort signal and a timeout. */
export interface RequestOptions {
	signal?: AbortSignal;
	timeoutMs?: number;
}

/** True when a request was aborted by its caller's signal (not by a timeout). */
export function isAbortError(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

export class ApiClient {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	/** WebSocket URL of an endpoint (ws:// or wss:// to match the API base). */
	socketUrl(endpoint: string): string {
		return `${this.baseUrl.replace(/^http/, 'ws')}${endpoint}`;
	}

	private async request<T>(
		endpoint: string,
		init: RequestInit,
		{ signal, timeoutMs }: RequestOptions = {}
	): Promise<T> {
		const url = `${this.baseUrl}${endpoint}`;
		const controller = new AbortController();
		const forwardAbort = () => controller.abort(signal?.reason);
		if (signal?.aborted) forwardAbort();
		signal?.addEventListener('abort', forwardAbort, { once: true });
		let timedOut = false;
		const timer =
			timeoutMs === undefined
				? undefined
				: setTimeout(() => {
						timedOut = true;
						controller.abort();
					}, timeoutMs);
		const headers =
			init.body instanceof FormData
				? init.headers
				: { 'Content-Type': 'application/json', ...init.headers };

		try {
			const response = await fetch(url, { ...init, headers, signal: controller.signal });

			if (!response.ok) {
				const body: ApiError = await response.json().catch(() => ({
					detail: `HTTP ${response.status}: ${response.statusText}`
				}));
				const message = formatErrorDetail(body?.detail) || `HTTP ${response.status}`;
				throw new ApiRequestError(message, response.status, body);
			}

			if (response.status === 204 || response.headers.get('content-length') === '0') {
				return undefined as T;
			}
			return await response.json();
		} catch (error) {
			if (timedOut && timeoutMs !== undefined) throw new ApiTimeoutError(timeoutMs);
			throw error;
		} finally {
			if (timer !== undefined) clearTimeout(timer);
			signal?.removeEventListener('abort', forwardAbort);
		}
	}

	async get<T>(endpoint: string, options?: RequestOptions): Promise<T> {
		return this.request<T>(endpoint, { method: 'GET' }, options);
	}

	async post<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> {
		return this.request<T>(
			endpoint,
			{ method: 'POST', body: data ? JSON.stringify(data) : undefined },
			options
		);
	}

	/** POST a multipart form (file uploads); the browser sets the content type. */
	async postForm<T>(endpoint: string, form: FormData, options?: RequestOptions): Promise<T> {
		return this.request<T>(endpoint, { method: 'POST', body: form }, options);
	}

	async put<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> {
		return this.request<T>(
			endpoint,
			{ method: 'PUT', body: data ? JSON.stringify(data) : undefined },
			options
		);
	}

	async delete<T>(endpoint: string, options?: RequestOptions): Promise<T> {
		return this.request<T>(endpoint, { method: 'DELETE' }, options);
	}
}

export const apiClient = new ApiClient();
