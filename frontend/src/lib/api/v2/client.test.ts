import { describe, expect, it, vi } from 'vitest';

import { ApiRequestError } from '../client';
import { createApi } from './client';

function respond(body: unknown, status = 200) {
	return vi.fn(
		async () =>
			new Response(JSON.stringify(body), {
				status,
				headers: { 'Content-Type': 'application/json' }
			})
	);
}

describe('the console API client', () => {
	it('returns the data of a successful call', async () => {
		const fetchImpl = respond({ status: 'ok', version: '2.1.16' });
		const api = createApi('http://console', fetchImpl);
		const { data } = await api.GET('/api/v2/health');
		expect(data).toEqual({ status: 'ok', version: '2.1.16' });
		const request = (fetchImpl.mock.calls[0] as unknown[])[0] as Request;
		expect(request.url).toBe('http://console/api/v2/health');
	});

	it('throws an ApiRequestError with the server reason', async () => {
		const api = createApi('http://console', respond({ detail: 'no run 01J' }, 404));
		const call = api.GET('/api/v2/jobs/{run_id}', { params: { path: { run_id: '01J' } } });
		await expect(call).rejects.toBeInstanceOf(ApiRequestError);
		await expect(call).rejects.toMatchObject({ status: 404, message: 'no run 01J' });
	});

	it("explains a refused command with the parser's problems", async () => {
		const detail = { problems: ["Missing argument 'OUT'."] };
		const api = createApi('http://console', respond({ detail }, 422));
		const call = api.POST('/api/v2/jobs', { body: { argv: ['image', 'image-from-ms'] } });
		await expect(call).rejects.toMatchObject({ status: 422, message: "Missing argument 'OUT'." });
	});
});
