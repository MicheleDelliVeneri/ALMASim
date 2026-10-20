import { describe, expect, it } from 'vitest';

import { selectionStats } from './selectionStats';

describe('selectionStats', () => {
	it('returns null for an empty selection', () => {
		expect(selectionStats([])).toBeNull();
	});

	it('computes mean, population std, sum, min and max', () => {
		const stats = selectionStats([1, 2, 3, 4]);
		expect(stats).toEqual({ mean: 2.5, std: Math.sqrt(1.25), sum: 10, min: 1, max: 4, count: 4 });
	});

	it('handles negative values', () => {
		expect(selectionStats([-3, -1])).toMatchObject({ min: -3, max: -1, mean: -2 });
	});

	it('does not overflow the call stack on large selections', () => {
		const values = new Float64Array(1_000_000).map((_, i) => i % 1000);
		const stats = selectionStats(values);
		expect(stats).toMatchObject({ min: 0, max: 999, count: 1_000_000 });
	});
});
