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

export class ApiClient {
	private baseUrl: string;

	constructor(baseUrl: string = API_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
		const url = `${this.baseUrl}${endpoint}`;
		const response = await fetch(url, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...options.headers
			}
		});

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
		return response.json();
	}

	async get<T>(endpoint: string): Promise<T> {
		return this.request<T>(endpoint, { method: 'GET' });
	}

	async post<T>(endpoint: string, data?: unknown): Promise<T> {
		return this.request<T>(endpoint, {
			method: 'POST',
			body: data ? JSON.stringify(data) : undefined
		});
	}

	async put<T>(endpoint: string, data?: unknown): Promise<T> {
		return this.request<T>(endpoint, {
			method: 'PUT',
			body: data ? JSON.stringify(data) : undefined
		});
	}

	async delete<T>(endpoint: string): Promise<T> {
		return this.request<T>(endpoint, { method: 'DELETE' });
	}
}

export const apiClient = new ApiClient();
