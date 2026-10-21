import { describe, expect, it } from 'vitest';

import { splitChunk, visibleWindow } from './logLines';

describe('splitChunk', () => {
	it('keeps an unfinished line for the next chunk', () => {
		const first = splitChunk('', 'one\ntwo\nthr');
		expect(first).toEqual({ lines: ['one', 'two'], partial: 'thr' });
		expect(splitChunk(first.partial, 'ee\nfour\n')).toEqual({
			lines: ['three', 'four'],
			partial: ''
		});
	});

	it('drops carriage returns and handles empty chunks', () => {
		expect(splitChunk('', 'a\r\nb\r\n')).toEqual({ lines: ['a', 'b'], partial: '' });
		expect(splitChunk('x', '')).toEqual({ lines: [], partial: 'x' });
	});
});

describe('visibleWindow', () => {
	it('covers the view plus the overscan, inside the log', () => {
		expect(visibleWindow(0, 400, 20, 10_000, 10)).toEqual({ start: 0, end: 30 });
		expect(visibleWindow(2000, 400, 20, 10_000, 10)).toEqual({ start: 90, end: 130 });
		expect(visibleWindow(199_800, 400, 20, 10_000, 10)).toEqual({ start: 9980, end: 10_000 });
	});

	it('is empty for an empty log', () => {
		expect(visibleWindow(0, 400, 20, 0)).toEqual({ start: 0, end: 0 });
	});
});
