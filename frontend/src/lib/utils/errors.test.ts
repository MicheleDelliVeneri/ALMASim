import { describe, expect, it } from 'vitest';

import { ApiRequestError, ApiTimeoutError } from '#lib/api/client.js';
import { errorMessage } from './errors';

describe('errorMessage', () => {
	it('uses the server reason of an API error', () => {
		expect(errorMessage(new ApiRequestError('no run 01J', 404, {}))).toBe('no run 01J');
	});

	it('points a 401 at the link almasim serve printed', () => {
		expect(errorMessage(new ApiRequestError('Not authenticated', 401, {}))).toMatch(
			/almasim serve/
		);
	});

	it('says when the server cannot be reached', () => {
		expect(errorMessage(new TypeError('Failed to fetch'))).toBe('Cannot reach the ALMASim server.');
	});

	it('handles timeouts, plain errors, strings and anything else', () => {
		expect(errorMessage(new ApiTimeoutError(30000))).toBe('Timed out after 30 s');
		expect(errorMessage(new Error('disk full'))).toBe('disk full');
		expect(errorMessage('just text')).toBe('just text');
		expect(errorMessage({ odd: true })).toBe('Something went wrong.');
	});
});
