import { describe, expect, it } from 'vitest';

import { formatBytes, formatDate, formatSpeed } from './format';

describe('formatBytes', () => {
	it.each([
		[0, '0 B'],
		[1023, '1023 B'],
		[1024, '1.0 KB'],
		[1536, '1.5 KB'],
		[5 * 1024 ** 2, '5.0 MB'],
		[3.5 * 1024 ** 3, '3.50 GB'],
		[2 * 1024 ** 4, '2.00 TB'],
		[3000 * 1024 ** 4, '3000.00 TB']
	])('formats %d as %s', (bytes, expected) => {
		expect(formatBytes(bytes)).toBe(expected);
	});

	it('rounds fractional bytes', () => {
		expect(formatBytes(12.6)).toBe('13 B');
	});
});

describe('formatSpeed', () => {
	it('appends /s', () => {
		expect(formatSpeed(2048)).toBe('2.0 KB/s');
		expect(formatSpeed(0.4)).toBe('0 B/s');
	});
});

describe('formatDate', () => {
	it('reads numbers as Unix seconds', () => {
		expect(formatDate(0)).toBe(new Date(0).toLocaleString());
	});

	it('formats ISO strings and passes other text through', () => {
		const iso = '2026-10-20T01:00:00Z';
		expect(formatDate(iso)).toBe(new Date(iso).toLocaleString());
		expect(formatDate('not a date')).toBe('not a date');
	});
});
