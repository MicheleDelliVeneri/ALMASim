import { afterEach, describe, expect, it, vi } from 'vitest';

import {
	ApiClient,
	ApiRequestError,
	ApiTimeoutError,
	formatErrorDetail,
	isAbortError
} from './client';

describe('formatErrorDetail', () => {
	it('keeps string details', () => {
		expect(formatErrorDetail('Download job not found')).toBe('Download job not found');
	});

	it('turns FastAPI validation errors into readable text', () => {
		const detail = [
			{ loc: ['body', 'destination'], msg: 'Field required', type: 'missing' },
			{ loc: ['query', 'limit'], msg: 'Input should be a valid integer', type: 'int_parsing' }
		];
		expect(formatErrorDetail(detail)).toBe(
			'destination: Field required; query.limit: Input should be a valid integer'
		);
	});

	it('falls back to JSON for other shapes and to "" for nothing', () => {
		expect(formatErrorDetail({ code: 7 })).toBe('{"code":7}');
		expect(formatErrorDetail(undefined)).toBe('');
	});
});

describe('ApiClient errors', () => {
	afterEach(() => vi.unstubAllGlobals());

	it('throws ApiRequestError with a readable message for a 422', async () => {
		const body = { detail: [{ loc: ['body', 'n_pix'], msg: 'Input should be > 0' }] };
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => new Response(JSON.stringify(body), { status: 422 }))
		);
		const error = (await new ApiClient('http://api')
			.post('/x', {})
			.catch((e: unknown) => e)) as ApiRequestError;
		expect(error).toBeInstanceOf(ApiRequestError);
		expect(error.message).toBe('n_pix: Input should be > 0');
		expect(error.message).not.toContain('[object Object]');
		expect(error.status).toBe(422);
		expect(error.body).toEqual(body);
	});

	it('uses the HTTP status when the body is not JSON', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => new Response('oops', { status: 502, statusText: 'Bad Gateway' }))
		);
		const error = (await new ApiClient('http://api')
			.get('/x')
			.catch((e: unknown) => e)) as ApiRequestError;
		expect(error.message).toBe('HTTP 502: Bad Gateway');
		expect(error.status).toBe(502);
	});
});

describe('ApiClient request options', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.useRealTimers();
	});

	/** A fetch that never answers until its signal aborts. */
	function hangingFetch() {
		return vi.fn(
			(_url: string, init: RequestInit) =>
				new Promise<Response>((_resolve, reject) => {
					init.signal?.addEventListener('abort', () =>
						reject(new DOMException('aborted', 'AbortError'))
					);
				})
		);
	}

	it('fails with ApiTimeoutError after timeoutMs', async () => {
		vi.useFakeTimers();
		vi.stubGlobal('fetch', hangingFetch());
		const pending = new ApiClient('http://api').get('/slow', { timeoutMs: 5000 }).catch((e) => e);
		await vi.advanceTimersByTimeAsync(5000);
		const error = await pending;
		expect(error).toBeInstanceOf(ApiTimeoutError);
		expect((error as ApiTimeoutError).timeoutMs).toBe(5000);
	});

	it("passes the caller's abort through as an AbortError", async () => {
		vi.stubGlobal('fetch', hangingFetch());
		const controller = new AbortController();
		const pending = new ApiClient('http://api')
			.get('/slow', { signal: controller.signal, timeoutMs: 60_000 })
			.catch((e) => e);
		controller.abort();
		const error = await pending;
		expect(isAbortError(error)).toBe(true);
		expect(error).not.toBeInstanceOf(ApiTimeoutError);
	});

	it('posts a form without a JSON content type', async () => {
		const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true })));
		vi.stubGlobal('fetch', fetchMock);
		const form = new FormData();
		form.append('method', 'sum');
		await new ApiClient('http://api').postForm('/integrate', form);
		const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];
		expect(init.body).toBe(form);
		expect(init.headers).toBeUndefined();
	});

	it('builds WebSocket URLs from the API base', () => {
		expect(new ApiClient('http://api:8000').socketUrl('/ws')).toBe('ws://api:8000/ws');
		expect(new ApiClient('https://almasim.example').socketUrl('/ws')).toBe(
			'wss://almasim.example/ws'
		);
	});
});
