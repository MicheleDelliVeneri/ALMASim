import { describe, expect, it } from 'vitest';

import { colorAt, percentileRange, renderPlane, stretch } from './imageRender';

describe('colours and stretches', () => {
	it('runs a map from its first to its last stop', () => {
		expect(colorAt('grey', 0)).toEqual([0, 0, 0]);
		expect(colorAt('grey', 1)).toEqual([255, 255, 255]);
		expect(colorAt('grey', 0.5)).toEqual([128, 128, 128]);
		expect(colorAt('skao', 0.5)).toEqual([229, 13, 127]);
		expect(colorAt('grey', 7)).toEqual([255, 255, 255]); // clipped
	});

	it('stretches keep 0 and 1 and lift the faint end', () => {
		for (const kind of ['linear', 'sqrt', 'log', 'asinh'] as const) {
			expect(stretch(0, kind)).toBeCloseTo(0);
			expect(stretch(1, kind)).toBeCloseTo(1);
		}
		expect(stretch(0.1, 'sqrt')).toBeGreaterThan(0.1);
		expect(stretch(0.01, 'log')).toBeGreaterThan(stretch(0.01, 'sqrt'));
	});
});

describe('percentileRange', () => {
	it('clips the extremes and ignores blanked pixels', () => {
		const plane = [Array.from({ length: 1001 }, (_, i) => i), [null, NaN as unknown as number]];
		expect(percentileRange(plane, 0.5, 99.5)).toEqual([5, 995]);
		expect(percentileRange([[null]])).toBeNull();
	});

	it('falls back to every value when the clip collapses, as on a point source', () => {
		const sparse = [Array.from({ length: 1000 }, () => 0), [0, 0, 7]];
		expect(percentileRange(sparse)).toEqual([0, 7]);
		expect(percentileRange([[3, 3]])).toEqual([3, 4]);
	});
});

describe('renderPlane', () => {
	it('draws FITS rows bottom up, blanked pixels transparent', () => {
		// Bottom row (first in FITS): 0, 1; top row: null, 1.
		const rgba = renderPlane(
			[
				[0, 1],
				[null, 1]
			],
			{ min: 0, max: 1, stretch: 'linear', colorMap: 'grey' }
		);
		expect(Array.from(rgba)).toEqual([
			0, 0, 0, 0, /* top-left: blank */ 255, 255, 255, 255,
			0, 0, 0, 255, /* bottom row */ 255, 255, 255, 255
		]); // prettier-ignore
	});
});
