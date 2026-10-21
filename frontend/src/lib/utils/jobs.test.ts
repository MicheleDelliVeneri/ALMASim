import { describe, expect, it } from 'vitest';

import type { Run } from '#lib/api/v2/client.js';
import {
	commandLine,
	elapsedLabel,
	filterRuns,
	formatDuration,
	shellQuote,
	whereLabel
} from './jobs';

function run(overrides: Partial<Run> = {}): Run {
	return {
		id: '01JRUN0000000000000000000A',
		short_id: '000000000A',
		owner: 'astro',
		command: 'image image-from-ms',
		argv: ['image', 'image-from-ms', 'params.csv', 'out'],
		title: '',
		executor: 'slurm',
		resources: {
			partition: 'zwickyA-CPU',
			account: null,
			cpus: 4,
			memory: '16G',
			time: '2-00:00:00'
		},
		params: {},
		parents: [],
		slurm_job_id: '6402',
		state: 'running',
		finished: false,
		reason: '',
		created_at: 1000,
		started_at: 1060,
		finished_at: null,
		host: 'node47',
		exit_code: null,
		signal: null,
		run_dir: '/runs/01JRUN',
		...overrides
	};
}

describe('formatDuration', () => {
	it('reads at a glance', () => {
		expect(formatDuration(45)).toBe('45 s');
		expect(formatDuration(12 * 60 + 5)).toBe('12 m');
		expect(formatDuration(3 * 3600 + 12 * 60)).toBe('3 h 12 m');
		expect(formatDuration(2 * 86400 + 4 * 3600)).toBe('2 d 4 h');
		expect(formatDuration(-3)).toBe('0 s');
	});
});

describe('elapsedLabel', () => {
	it('counts from the start, to the end or now', () => {
		expect(elapsedLabel(run(), 1060 + 125)).toBe('2 m');
		expect(elapsedLabel(run({ finished: true, finished_at: 1060 + 30 }), 99_999)).toBe('30 s');
	});

	it('says how long a run has waited, and nothing for one that never started', () => {
		expect(elapsedLabel(run({ started_at: null, state: 'queued' }), 1000 + 300)).toBe(
			'waiting 5 m'
		);
		expect(elapsedLabel(run({ started_at: null, state: 'cancelled', finished: true }), 5000)).toBe(
			'—'
		);
	});
});

describe('commandLine', () => {
	it('quotes only what the shell would split or expand', () => {
		expect(shellQuote('--cube')).toBe('--cube');
		expect(shellQuote('/data/my cubes')).toBe("'/data/my cubes'");
		expect(shellQuote("it's")).toBe("'it'\\''s'");
		expect(shellQuote('$HOME')).toBe("'$HOME'");
		expect(commandLine(['metadata', '--band', '3', '--science-keyword', 'Galaxy evolution'])).toBe(
			"almasim metadata --band 3 --science-keyword 'Galaxy evolution'"
		);
	});
});

describe('whereLabel', () => {
	it('names the Slurm job and partition, or the console server', () => {
		expect(whereLabel(run())).toBe('Slurm 6402 · zwickyA-CPU');
		expect(whereLabel(run({ slurm_job_id: null }))).toBe('Slurm · zwickyA-CPU');
		expect(whereLabel(run({ executor: 'server', slurm_job_id: null }))).toBe('console server');
	});
});

describe('filterRuns', () => {
	const runs = [
		run({ id: 'A', owner: 'astro', state: 'running' }),
		run({ id: 'B', owner: 'astro', state: 'failed', finished: true, command: 'metadata' }),
		run({ id: 'C', owner: 'teammate', state: 'succeeded', finished: true }),
		run({
			id: 'D',
			owner: 'teammate',
			state: 'lost',
			finished: true,
			title: 'noise rerun',
			executor: 'server',
			slurm_job_id: null
		})
	];
	const ids = (out: Run[]) => out.map((r) => r.id);
	const base = { owner: 'team', user: 'astro', state: 'all', query: '' } as const;

	it("keeps your runs, or the team's", () => {
		expect(ids(filterRuns(runs, { ...base, owner: 'mine' }))).toEqual(['A', 'B']);
		expect(ids(filterRuns(runs, base))).toEqual(['A', 'B', 'C', 'D']);
		// Before the console says who you are, "mine" shows everything.
		expect(ids(filterRuns(runs, { ...base, owner: 'mine', user: null }))).toHaveLength(4);
	});

	it('filters by state', () => {
		expect(ids(filterRuns(runs, { ...base, state: 'active' }))).toEqual(['A']);
		expect(ids(filterRuns(runs, { ...base, state: 'failed' }))).toEqual(['B', 'D']);
		expect(ids(filterRuns(runs, { ...base, state: 'succeeded' }))).toEqual(['C']);
	});

	it('matches every word of the search anywhere in the run', () => {
		expect(ids(filterRuns(runs, { ...base, query: 'metadata' }))).toEqual(['B']);
		expect(ids(filterRuns(runs, { ...base, query: 'NOISE teammate' }))).toEqual(['D']);
		expect(ids(filterRuns(runs, { ...base, query: 'params.csv 6402' }))).toHaveLength(3);
	});
});
