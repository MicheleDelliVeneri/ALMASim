/**
 * Every page's view of the runs, kept live by one server-sent-events stream
 * (`GET /api/v2/events`): a snapshot of the latest runs when it connects,
 * then one event per state change. The browser reconnects by itself and
 * sends the last event number it saw, so a dropped connection only misses
 * what the server can no longer replay, and then the server sends a fresh
 * snapshot. Pages read `jobs.list` or `jobs.get(id)`; they never poll.
 */
import { API_BASE_URL } from '#lib/config.js';
import type { Run } from '#lib/api/v2/client.js';

/** What the store needs of an EventSource (tests pass a fake). */
export interface EventStream {
	addEventListener(type: string, listener: (event: MessageEvent) => void): void;
	close(): void;
	onopen: ((event: Event) => void) | null;
	onerror: ((event: Event) => void) | null;
	/** 2 (CLOSED) once the browser has given up reconnecting. */
	readonly readyState: number;
}

export type Connect = (url: string) => EventStream;

const browserConnect: Connect = (url) => new EventSource(url) as unknown as EventStream;

const CLOSED = 2;
const RETRY_MS = [2000, 5000, 15000, 60000];

export class JobsStore {
	/** The runs by id. */
	runs = $state<Record<string, Run>>({});
	/** Who this console runs as (from the snapshot), to tell your runs from the team's. */
	user = $state<string | null>(null);
	connected = $state(false);
	/** True once the first snapshot has arrived. */
	loaded = $state(false);

	/** Newest first. */
	list = $derived(Object.values(this.runs).sort((a, b) => b.created_at - a.created_at));
	active = $derived(this.list.filter((run) => !run.finished));

	#connect: Connect;
	#url: string;
	#stream: EventStream | null = null;
	#retries = 0;
	#retryTimer: ReturnType<typeof setTimeout> | undefined;

	constructor(connect: Connect = browserConnect, url = `${API_BASE_URL}/api/v2/events`) {
		this.#connect = connect;
		this.#url = url;
	}

	/** Open the stream (once; later calls do nothing while it is open). */
	start(): void {
		if (this.#stream) return;
		const stream = this.#connect(this.#url);
		stream.onopen = () => {
			this.connected = true;
			this.#retries = 0;
		};
		stream.onerror = () => {
			this.connected = false;
			// The browser retries a dropped stream by itself, but gives up on an
			// error response (a restarting server behind a proxy): start over, slowly.
			if (stream.readyState === CLOSED && this.#stream === stream) {
				this.#stream = null;
				const delay = RETRY_MS[Math.min(this.#retries++, RETRY_MS.length - 1)];
				this.#retryTimer = setTimeout(() => this.start(), delay);
			}
		};
		stream.addEventListener('snapshot', (event) => {
			const { runs, user } = JSON.parse(event.data) as { runs: Run[]; user: string };
			this.runs = Object.fromEntries(runs.map((run) => [run.id, run]));
			this.user = user;
			this.loaded = true;
			this.connected = true;
		});
		stream.addEventListener('run', (event) => {
			const { run } = JSON.parse(event.data) as { run: Run };
			this.put(run);
		});
		this.#stream = stream;
	}

	stop(): void {
		clearTimeout(this.#retryTimer);
		this.#stream?.close();
		this.#stream = null;
		this.connected = false;
	}

	get(id: string): Run | undefined {
		return this.runs[id];
	}

	/** A run a page just changed (submitted, cancelled), ahead of its event. */
	put(run: Run): void {
		this.runs[run.id] = run;
	}
}

export const jobs = new JobsStore();
