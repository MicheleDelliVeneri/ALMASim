/**
 * An image plane (rows of values, null for blanked pixels) as RGBA pixels:
 * clip to a range, stretch, then colour. Pure, so the viewer only draws.
 */

export type Plane = (number | null)[][];
export type Stretch = 'linear' | 'sqrt' | 'log' | 'asinh';
export type ColorMap = 'grey' | 'viridis' | 'inferno' | 'skao';

export const STRETCHES: Stretch[] = ['linear', 'sqrt', 'log', 'asinh'];
export const COLOR_MAPS: ColorMap[] = ['inferno', 'viridis', 'grey', 'skao'];

type Rgb = [number, number, number];

// Stops from 0 to 1, interpolated linearly between.
const STOPS: Record<ColorMap, Rgb[]> = {
	grey: [
		[0, 0, 0],
		[255, 255, 255]
	],
	viridis: [
		[68, 1, 84],
		[59, 82, 139],
		[33, 145, 140],
		[94, 201, 98],
		[253, 231, 37]
	],
	inferno: [
		[0, 0, 4],
		[87, 16, 110],
		[188, 55, 84],
		[249, 142, 9],
		[252, 255, 164]
	],
	// The SKAO palette: navy, magenta, white.
	skao: [
		[10, 30, 70],
		[229, 13, 127],
		[255, 255, 255]
	]
};

/** The colour at `t` in [0, 1]. */
export function colorAt(map: ColorMap, t: number): Rgb {
	const stops = STOPS[map];
	const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
	const i = Math.min(stops.length - 2, Math.floor(x));
	const f = x - i;
	const [a, b] = [stops[i], stops[i + 1]];
	return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f)) as Rgb;
}

/** `t` in [0, 1] after the stretch, still in [0, 1]. */
export function stretch(t: number, kind: Stretch): number {
	const x = Math.min(1, Math.max(0, t));
	switch (kind) {
		case 'sqrt':
			return Math.sqrt(x);
		case 'log':
			return Math.log10(1 + 999 * x) / 3;
		case 'asinh':
			return Math.asinh(10 * x) / Math.asinh(10);
		default:
			return x;
	}
}

/** The values at the `lo` and `hi` percentiles (0–100) of the finite pixels; null when none. */
export function percentileRange(plane: Plane, lo = 0.5, hi = 99.5): [number, number] | null {
	const values: number[] = [];
	for (const row of plane)
		for (const v of row) if (v !== null && Number.isFinite(v)) values.push(v);
	if (!values.length) return null;
	values.sort((a, b) => a - b);
	const at = (p: number) =>
		values[Math.min(values.length - 1, Math.max(0, Math.round((p / 100) * (values.length - 1))))];
	const [low, high] = [at(lo), at(hi)];
	if (low < high) return [low, high];
	// A sparse image (a point source on zeros): the clip collapses, so take every value.
	const [first, last] = [values[0], values[values.length - 1]];
	return first < last ? [first, last] : [first, first + 1];
}

export interface RenderOptions {
	min: number;
	max: number;
	stretch: Stretch;
	colorMap: ColorMap;
}

/**
 * RGBA pixels, row by row as an ImageData wants them. FITS rows run bottom
 * to top, so the last row is drawn first. Blanked pixels are transparent.
 */
export function renderPlane(plane: Plane, options: RenderOptions): Uint8ClampedArray<ArrayBuffer> {
	const height = plane.length;
	const width = height ? plane[0].length : 0;
	const out = new Uint8ClampedArray(new ArrayBuffer(width * height * 4));
	const span = options.max - options.min || 1;
	for (let y = 0; y < height; y++) {
		const row = plane[height - 1 - y];
		for (let x = 0; x < width; x++) {
			const v = row[x];
			const at = (y * width + x) * 4;
			if (v === null || !Number.isFinite(v)) continue; // alpha stays 0
			const [r, g, b] = colorAt(
				options.colorMap,
				stretch((v - options.min) / span, options.stretch)
			);
			out[at] = r;
			out[at + 1] = g;
			out[at + 2] = b;
			out[at + 3] = 255;
		}
	}
	return out;
}
