import { describe, expect, it } from 'vitest';
import {
	fitScale,
	imageToScreen,
	screenToImage,
	selectionValues,
	type CanvasView
} from './canvasGeometry';

const view: CanvasView = {
	displayWidth: 400,
	displayHeight: 200,
	imageWidth: 100,
	imageHeight: 100,
	scale: 2,
	panX: 30,
	panY: -10
};

describe('canvasGeometry', () => {
	it('fits the image in the display', () => {
		expect(fitScale(view)).toBe(2);
	});

	it('maps screen and image points both ways', () => {
		const point = { x: 123.5, y: 77.25 };
		const back = imageToScreen(view, screenToImage(view, point));
		expect(back.x).toBeCloseTo(point.x);
		expect(back.y).toBeCloseTo(point.y);
		// The display centre, shifted by the pan, is the image centre.
		expect(screenToImage(view, { x: 230, y: 90 })).toEqual({ x: 50, y: 50 });
	});

	it('keeps a selection on its pixels when the view changes', () => {
		const corner = screenToImage(view, { x: 250, y: 120 });
		const zoomed = { ...view, scale: 4, panX: -50 };
		const moved = imageToScreen(zoomed, corner);
		expect(screenToImage(zoomed, moved).x).toBeCloseTo(corner.x);
		expect(moved.x).not.toBeCloseTo(250);
	});

	it('collects the pixels of a rectangle or a circle', () => {
		const image = Array.from({ length: 5 }, (_, y) =>
			Array.from({ length: 5 }, (_, x) => 10 * y + x)
		);
		expect(selectionValues(image, 'rectangle', { x: 1.7, y: 1.2 }, { x: 3.1, y: 2.9 })).toEqual([
			11, 12, 13, 21, 22, 23
		]);
		expect(selectionValues(image, 'rectangle', { x: -3, y: -3 }, { x: 0, y: 0 })).toEqual([0]);
		expect(selectionValues(image, 'circle', { x: 1, y: 1 }, { x: 3, y: 3 }).sort()).toEqual([
			11, 12, 13, 21, 22, 23, 31, 32, 33
		]);
	});
});
