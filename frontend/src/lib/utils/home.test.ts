import { describe, expect, it } from 'vitest';

import type { Run } from '#lib/api/v2/client.js';
import { homeSummary } from './home';

let n = 0;
function run(overrides: Partial<Run>): Run {
	n += 1;
	const finished = ['succeeded', 'failed', 'cancelled', 'lost'].includes(
		overrides.state ?? 'running'
	);
	return {
		id: `RUN${n}`,
		short_id: `R${n}`,
		owner: 'astro',
		command: 'doctor',
		argv: ['doctor'],
		title: '',
		executor: 'slurm',
		resources: { partition: null, account: null, cpus: 1, memory: '4G', time: '1:00:00' },
		params: {},
		parents: [],
		slurm_job_id: null,
		state: 'running',
		finished,
		reason: '',
		created_at: 1000,
		started_at: 1000,
		finished_at: finished ? 2000 : null,
		host: null,
		exit_code: null,
		signal: null,
		run_dir: `/runs/${n}`,
		...overrides
	};
}

describe('homeSummary', () => {
	const NOW = 2000 + 3600;

	it('says what is running and what needs attention', () => {
		const runs = [
			run({ state: 'running' }),
			run({ state: 'running' }),
			run({ state: 'queued' }),
			run({ state: 'failed' }),
			run({ state: 'succeeded' }),
			run({ owner: 'teammate', state: 'running' })
		];
		const s = homeSummary(runs, 'astro', NOW);
		expect(s.headline).toBe('2 jobs running, 1 waiting, 1 needs your attention');
		expect(s.mine.map((r) => r.state)).toEqual([
			'running',
			'running',
			'queued',
			'failed',
			'succeeded'
		]);
		expect(s.team.map((r) => r.owner)).toEqual(['teammate']);
		expect(s.results).toHaveLength(1);
	});

	it('forgets failures older than a day, and words a quiet day', () => {
		const old = run({ state: 'failed', finished_at: 2000 });
		expect(homeSummary([old], 'astro', 2000 + 2 * 86_400).headline).toBe('Nothing running');
		expect(homeSummary([run({ state: 'queued' })], 'astro', NOW).headline).toBe('1 job waiting');
	});
});
