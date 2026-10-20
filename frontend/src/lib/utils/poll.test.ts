import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createPoller } from './poll';

describe('createPoller', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('polls while the tick reports active work, then stops', async () => {
		const results = [true, true, false];
		const tick = vi.fn(async () => results.shift() ?? false);
		const poller = createPoller(tick, 1000);
		poller.start();
		await vi.advanceTimersByTimeAsync(5000);
		expect(tick).toHaveBeenCalledTimes(3);
		expect(poller.running).toBe(false);
	});

	it('start is a no-op while running and restarts after it stopped', async () => {
		const tick = vi.fn(async () => false);
		const poller = createPoller(tick, 1000);
		poller.start();
		poller.start();
		await vi.advanceTimersByTimeAsync(1000);
		expect(tick).toHaveBeenCalledTimes(1);
		poller.start();
		await vi.advanceTimersByTimeAsync(1000);
		expect(tick).toHaveBeenCalledTimes(2);
	});

	it('stop cancels the next tick', async () => {
		const tick = vi.fn(async () => true);
		const poller = createPoller(tick, 1000);
		poller.start();
		await vi.advanceTimersByTimeAsync(1000);
		poller.stop();
		await vi.advanceTimersByTimeAsync(5000);
		expect(tick).toHaveBeenCalledTimes(1);
	});

	it('a failing tick stops polling', async () => {
		const tick = vi.fn(async () => {
			throw new Error('offline');
		});
		const poller = createPoller(tick, 1000);
		poller.start();
		await vi.advanceTimersByTimeAsync(3000);
		expect(tick).toHaveBeenCalledTimes(1);
		expect(poller.running).toBe(false);
	});
});
