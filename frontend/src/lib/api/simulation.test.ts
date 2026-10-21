import { describe, expect, it } from 'vitest';

import { parseStatusMessage } from './simulation';

describe('parseStatusMessage', () => {
	it('returns a status message as it is', () => {
		const status = parseStatusMessage('{"status":"running","progress":42}');
		expect(status.status).toBe('running');
		expect(status.progress).toBe(42);
	});

	it("turns the server's not-found message into an error", () => {
		expect(() => parseStatusMessage('{"error":"Simulation not found"}')).toThrow(
			'Status unavailable: Simulation not found'
		);
	});

	it('rejects messages without a status', () => {
		expect(() => parseStatusMessage('{"progress":1}')).toThrow('malformed status message');
		expect(() => parseStatusMessage('not json')).toThrow();
	});
});
