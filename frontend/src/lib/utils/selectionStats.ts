export interface SelectionStats {
	mean: number;
	std: number;
	sum: number;
	min: number;
	max: number;
	count: number;
}

/**
 * Mean, population standard deviation, sum, min and max of a selection's pixels.
 *
 * Min and max are found in a loop: `Math.min(...values)` passes every pixel as
 * an argument and throws a RangeError for selections of a few hundred thousand
 * pixels.
 */
export function selectionStats(values: ArrayLike<number>): SelectionStats | null {
	const count = values.length;
	if (count === 0) return null;

	let sum = 0;
	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i < count; i++) {
		const value = values[i];
		sum += value;
		if (value < min) min = value;
		if (value > max) max = value;
	}
	const mean = sum / count;
	let squares = 0;
	for (let i = 0; i < count; i++) {
		squares += (values[i] - mean) ** 2;
	}
	return { mean, std: Math.sqrt(squares / count), sum, min, max, count };
}
