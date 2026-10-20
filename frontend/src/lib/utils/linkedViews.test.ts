import { describe, expect, it } from 'vitest';
import { INITIAL_VIEW, applyView } from './linkedViews';

const panels = [
	{ id: 'a', scale: 1, panX: 0, panY: 0, title: 'A' },
	{ id: 'b', scale: 2, panX: 5, panY: 5, title: 'B' }
];

describe('linkedViews', () => {
	it('changes only the panel when not linked', () => {
		const next = applyView(panels, 'a', { scale: 3 }, false);
		expect(next.map((p) => p.scale)).toEqual([3, 2]);
		expect(next[1]).toBe(panels[1]);
		expect(next[0].title).toBe('A');
	});

	it('changes every panel when linked', () => {
		const next = applyView(panels, 'a', { panX: 7, panY: -1 }, true);
		expect(next.map((p) => [p.panX, p.panY])).toEqual([
			[7, -1],
			[7, -1]
		]);
	});

	it('resets to the initial view', () => {
		const next = applyView(panels, 'b', INITIAL_VIEW, false);
		expect(next[1]).toMatchObject({ scale: 1, panX: 0, panY: 0 });
	});
});
