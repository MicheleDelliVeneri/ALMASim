import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiClient, ApiRequestError, formatErrorDetail } from './client';

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
