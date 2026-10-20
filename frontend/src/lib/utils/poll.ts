/**
 * A poller that runs `tick` every `intervalMs` while it reports active work.
 *
 * `tick` resolves to whether to keep polling. Ticks never overlap: the next
 * one is scheduled after the previous one settles. `start()` is a no-op while
 * the poller is running, so callers may call it after any action that may
 * have started work.
 */
export interface Poller {
	start(): void;
	stop(): void;
	readonly running: boolean;
}

export function createPoller(tick: () => Promise<boolean>, intervalMs: number): Poller {
	let timer: ReturnType<typeof setTimeout> | undefined;
	let running = false;

	async function run() {
		timer = undefined;
		let again: boolean;
		try {
			again = await tick();
		} catch {
			again = false;
		}
		if (running && again) {
			timer = setTimeout(run, intervalMs);
		} else {
			running = false;
		}
	}

	return {
		start() {
			if (running) return;
			running = true;
			timer = setTimeout(run, intervalMs);
		},
		stop() {
			running = false;
			if (timer !== undefined) clearTimeout(timer);
			timer = undefined;
		},
		get running() {
			return running;
		}
	};
}
