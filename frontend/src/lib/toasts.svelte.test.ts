import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Toasts } from './toasts.svelte';

describe('Toasts', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('shows a notice and lets it fade', () => {
		const toasts = new Toasts();
		toasts.show('Job submitted', 'ok', 1000);
		expect(toasts.items.map((t) => t.message)).toEqual(['Job submitted']);
		vi.advanceTimersByTime(1000);
		expect(toasts.items).toEqual([]);
	});

	it('keeps errors until dismissed', () => {
		const toasts = new Toasts();
		const id = toasts.show('Submit failed', 'danger', 1000);
		vi.advanceTimersByTime(60_000);
		expect(toasts.items).toHaveLength(1);
		toasts.dismiss(id);
		expect(toasts.items).toEqual([]);
	});
});
