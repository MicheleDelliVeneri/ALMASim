/**
 * Zoom and pan of side-by-side image panels, optionally linked (MADIV-28 T7.5).
 */

export interface PanelView {
	id: string;
	scale: number;
	panX: number;
	panY: number;
}

export type ViewChange = Partial<Pick<PanelView, 'scale' | 'panX' | 'panY'>>;

/** The zoom and pan a panel starts with, and returns to on reset. */
export const INITIAL_VIEW = { scale: 1.0, panX: 0, panY: 0 } as const;

/** ``panels`` with ``change`` applied to panel ``id``, or to every panel when ``linked``. */
export function applyView<T extends PanelView>(
	panels: readonly T[],
	id: string,
	change: ViewChange,
	linked: boolean
): T[] {
	return panels.map((panel) => (linked || panel.id === id ? { ...panel, ...change } : panel));
}
