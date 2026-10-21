/**
 * A followed log arrives in byte-offset chunks that can end mid-line; these
 * turn chunks into whole lines, and pick the lines a scrolled view shows.
 */

/** Whole lines in `partial + text`, and the unfinished rest. */
export function splitChunk(partial: string, text: string): { lines: string[]; partial: string } {
	const parts = (partial + text).split('\n');
	const rest = parts.pop() ?? '';
	return { lines: parts.map((line) => line.replace(/\r$/, '')), partial: rest };
}

/**
 * The lines [start, end) to draw for a view scrolled to `scrollTop`, with
 * `overscan` extra lines each side so fast scrolling shows no gaps.
 */
export function visibleWindow(
	scrollTop: number,
	viewHeight: number,
	lineHeight: number,
	total: number,
	overscan = 30
): { start: number; end: number } {
	const first = Math.floor(Math.max(0, scrollTop) / lineHeight);
	const count = Math.ceil(viewHeight / lineHeight);
	return {
		start: Math.max(0, first - overscan),
		end: Math.min(total, first + count + overscan)
	};
}
