import { describe, expect, it } from 'vitest';
import {
	DEFAULT_COLUMN_ORDER,
	PINNED_COLUMNS,
	filterRowIndices,
	formatColumnName,
	matchesFilter,
	moveColumn,
	stringifyValue,
	virtualWindow
} from './resultsTable';

describe('resultsTable', () => {
	it('starts with the preferred columns and leaves the pinned ones out', () => {
		expect(DEFAULT_COLUMN_ORDER.slice(0, 2)).toEqual(['Band', 'Array_type']);
		for (const pinned of PINNED_COLUMNS) expect(DEFAULT_COLUMN_ORDER).not.toContain(pinned);
		expect(new Set(DEFAULT_COLUMN_ORDER).size).toBe(DEFAULT_COLUMN_ORDER.length);
	});

	it('labels known columns and title-cases the rest', () => {
		expect(formatColumnName('Ang.res.')).toBe('Ang. Res. (arcsec)');
		expect(formatColumnName('some_new_column')).toBe('Some New Column');
	});

	it('stringifies arrays, objects and missing values', () => {
		expect(stringifyValue(['a', 'b'])).toBe('a, b');
		expect(stringifyValue({ x: 1 })).toBe('{"x":1}');
		expect(stringifyValue(null)).toBe('');
		expect(stringifyValue(3)).toBe('3');
	});

	it('matches substrings, or globs over the whole cell', () => {
		expect(matchesFilter('2023.1.00133.S', '00133')).toBe(true);
		expect(matchesFilter('2023.1.00133.S', '2023*.s')).toBe(true);
		expect(matchesFilter('2023.1.00133.S', '2022*')).toBe(false);
		expect(matchesFilter('a(b', 'a(*')).toBe(true);
	});

	it('keeps the positions of the rows every active filter matches', () => {
		const rows = [
			{ Band: 3, proposal_id: '2023.1.1.S' },
			{ Band: 6, proposal_id: '2022.1.2.S' },
			{ Band: 6, proposal_id: '2023.1.3.S' }
		];
		expect(filterRowIndices(rows, { Band: '6', proposal_id: '2023*' })).toEqual([2]);
		expect(filterRowIndices(rows, { Band: '' })).toEqual([0, 1, 2]);
	});

	it('moves a column to the target position', () => {
		expect(moveColumn(['a', 'b', 'c'], 'a', 'c')).toEqual(['b', 'c', 'a']);
		expect(moveColumn(['a', 'b', 'c'], 'c', 'a')).toEqual(['c', 'a', 'b']);
		expect(moveColumn(['a', 'b'], 'x', 'a')).toEqual(['a', 'b']);
	});

	it('renders the visible rows plus the overscan, padding the rest', () => {
		expect(virtualWindow(0, 1000, 36, 480, 15)).toEqual({
			start: 0,
			end: 29,
			topPad: 0,
			bottomPad: (1000 - 29) * 36
		});
		const scrolled = virtualWindow(3600, 1000, 36, 480, 15);
		expect(scrolled.start).toBe(85);
		expect(scrolled.end).toBe(129);
		expect(virtualWindow(0, 3, 36, 480, 15)).toEqual({ start: 0, end: 3, topPad: 0, bottomPad: 0 });
	});
});
