/**
 * Geometry of the image canvas: mapping between screen and image pixels, and the
 * pixels a selection covers (MADIV-28 T1.14). Selections are kept in image
 * coordinates, so they stay on the same pixels when the view is zoomed or panned.
 */

export interface Point {
	x: number;
	y: number;
}

export interface CanvasView {
	displayWidth: number;
	displayHeight: number;
	imageWidth: number;
	imageHeight: number;
	scale: number;
	panX: number;
	panY: number;
}

export type SelectionShape = 'rectangle' | 'circle';

/** The zoom that fits the whole image in the display at scale 1. */
export function fitScale(view: CanvasView): number {
	return Math.min(view.displayWidth / view.imageWidth, view.displayHeight / view.imageHeight);
}

/** A screen point (CSS pixels in the canvas) in image pixel coordinates (not rounded). */
export function screenToImage(view: CanvasView, point: Point): Point {
	const total = view.scale * fitScale(view);
	return {
		x: (point.x - view.displayWidth / 2 - view.panX) / total + view.imageWidth / 2,
		y: (point.y - view.displayHeight / 2 - view.panY) / total + view.imageHeight / 2
	};
}

/** An image point on the screen: the inverse of {@link screenToImage}. */
export function imageToScreen(view: CanvasView, point: Point): Point {
	const total = view.scale * fitScale(view);
	return {
		x: (point.x - view.imageWidth / 2) * total + view.displayWidth / 2 + view.panX,
		y: (point.y - view.imageHeight / 2) * total + view.displayHeight / 2 + view.panY
	};
}

/** The values of the pixels inside a selection given in image coordinates. */
export function selectionValues(
	image: number[][],
	shape: SelectionShape,
	start: Point,
	end: Point
): number[] {
	const height = image.length;
	const width = image[0]?.length || 0;
	const a = { x: Math.floor(start.x), y: Math.floor(start.y) };
	const b = { x: Math.floor(end.x), y: Math.floor(end.y) };
	const values: number[] = [];
	if (shape === 'rectangle') {
		const x1 = Math.max(0, Math.min(a.x, b.x));
		const x2 = Math.min(width - 1, Math.max(a.x, b.x));
		const y1 = Math.max(0, Math.min(a.y, b.y));
		const y2 = Math.min(height - 1, Math.max(a.y, b.y));
		for (let y = y1; y <= y2; y++) {
			for (let x = x1; x <= x2; x++) values.push(image[y][x]);
		}
		return values;
	}
	const centerX = (a.x + b.x) / 2;
	const centerY = (a.y + b.y) / 2;
	const radius = Math.hypot(b.x - a.x, b.y - a.y) / 2;
	const x1 = Math.max(0, Math.floor(centerX - radius));
	const x2 = Math.min(width - 1, Math.ceil(centerX + radius));
	const y1 = Math.max(0, Math.floor(centerY - radius));
	const y2 = Math.min(height - 1, Math.ceil(centerY + radius));
	for (let y = y1; y <= y2; y++) {
		for (let x = x1; x <= x2; x++) {
			if (Math.hypot(x - centerX, y - centerY) <= radius) values.push(image[y][x]);
		}
	}
	return values;
}
