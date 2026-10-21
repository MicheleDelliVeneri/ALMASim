/**
 * The time, in seconds, for elapsed-time labels: one timer for every page,
 * running only while some page uses it (`clock.use()` in an effect).
 */
class Clock {
	now = $state(Date.now() / 1000);
	#users = 0;
	#timer: ReturnType<typeof setInterval> | undefined;

	/** Start ticking for a caller; the returned function stops it again. */
	use(): () => void {
		if (this.#users++ === 0) {
			this.now = Date.now() / 1000;
			this.#timer = setInterval(() => (this.now = Date.now() / 1000), 1000);
		}
		return () => {
			if (--this.#users === 0) clearInterval(this.#timer);
		};
	}
}

export const clock = new Clock();
