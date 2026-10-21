import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Run } from '#lib/api/v2/client.js';
import { JobsStore, type EventStream } from './jobs.svelte';

/** An EventSource stand-in the test drives by hand. */
class FakeStream implements EventStream {
	listeners: Record<string, ((event: MessageEvent) => void)[]> = {};
	onopen: ((event: Event) => void) | null = null;
	onerror: ((event: Event) => void) | null = null;
	readyState = 0;
	closed = false;

	addEventListener(type: string, listener: (event: MessageEvent) => void) {
		(this.listeners[type] ??= []).push(listener);
	}

	close() {
		this.closed = true;
		this.readyState = 2;
	}

	emit(type: string, data: unknown) {
		const event = new MessageEvent(type, { data: JSON.stringify(data) });
		for (const listener of this.listeners[type] ?? []) listener(event);
	}

	open() {
		this.readyState = 1;
		this.onopen?.(new Event('open'));
	}

	fail(closed: boolean) {
		this.readyState = closed ? 2 : 0;
		this.onerror?.(new Event('error'));
	}
}

function run(id: string, state: Run['state'], created_at: number): Run {
	const finished = ['succeeded', 'failed', 'cancelled', 'lost'].includes(state);
	return {
		id,
		short_id: id.slice(-10),
		owner: 'astro',
		command: 'doctor',
		argv: ['doctor'],
		title: '',
		executor: 'slurm',
		resources: { partition: null, account: null, cpus: 1, memory: '4G', time: '02:00:00' },
		params: {},
		parents: [],
		slurm_job_id: null,
		state,
		finished,
		reason: '',
		created_at,
		started_at: null,
		finished_at: null,
		host: null,
		exit_code: null,
		signal: null,
		run_dir: `/runs/${id}`
	};
}

describe('JobsStore', () => {
	let streams: FakeStream[];
	let store: JobsStore;

	beforeEach(() => {
		streams = [];
		store = new JobsStore((url) => {
			expect(url).toBe('/api/v2/events');
			const stream = new FakeStream();
			streams.push(stream);
			return stream;
		}, '/api/v2/events');
	});

	afterEach(() => {
		store.stop();
		vi.useRealTimers();
	});

	it('loads the snapshot, newest first, and follows state changes', () => {
		store.start();
		streams[0].open();
		streams[0].emit('snapshot', {
			user: 'astro',
			runs: [run('A', 'succeeded', 1), run('B', 'running', 2)]
		});
		expect(store.loaded).toBe(true);
		expect(store.user).toBe('astro');
		expect(store.list.map((r) => r.id)).toEqual(['B', 'A']);
		expect(store.active.map((r) => r.id)).toEqual(['B']);

		streams[0].emit('run', { type: 'run', run: run('B', 'failed', 2) });
		expect(store.get('B')?.state).toBe('failed');
		expect(store.active).toEqual([]);

		streams[0].emit('run', { type: 'run', run: run('C', 'queued', 3) });
		expect(store.list.map((r) => r.id)).toEqual(['C', 'B', 'A']);
	});

	it('opens one stream however often it is started', () => {
		store.start();
		store.start();
		expect(streams).toHaveLength(1);
	});

	it('replaces everything on a fresh snapshot', () => {
		store.start();
		streams[0].emit('snapshot', { user: 'astro', runs: [run('A', 'running', 1)] });
		streams[0].emit('snapshot', { user: 'astro', runs: [run('B', 'queued', 2)] });
		expect(Object.keys(store.runs)).toEqual(['B']);
		streams[0].emit('gone', { type: 'gone', id: 'B' });
		expect(store.get('B')).toBeUndefined();
		expect(store.list).toEqual([]);
	});

	it('takes a run a page just changed', () => {
		store.put(run('D', 'queued', 4));
		expect(store.get('D')?.state).toBe('queued');
	});

	it('leaves reconnecting to the browser while it still tries', () => {
		vi.useFakeTimers();
		store.start();
		streams[0].open();
		streams[0].fail(false);
		expect(store.connected).toBe(false);
		vi.advanceTimersByTime(120_000);
		expect(streams).toHaveLength(1);
	});

	it('starts over, more slowly each time, once the browser gives up', () => {
		vi.useFakeTimers();
		store.start();
		streams[0].fail(true);
		vi.advanceTimersByTime(1999);
		expect(streams).toHaveLength(1);
		vi.advanceTimersByTime(1);
		expect(streams).toHaveLength(2);
		streams[1].fail(true);
		vi.advanceTimersByTime(4999);
		expect(streams).toHaveLength(2);
		vi.advanceTimersByTime(1);
		expect(streams).toHaveLength(3);
		// A good connection resets the delay.
		streams[2].open();
		streams[2].fail(true);
		vi.advanceTimersByTime(2000);
		expect(streams).toHaveLength(4);
	});

	it('stops reconnecting when stopped', () => {
		vi.useFakeTimers();
		store.start();
		streams[0].fail(true);
		store.stop();
		vi.advanceTimersByTime(120_000);
		expect(streams).toHaveLength(1);
	});
});
