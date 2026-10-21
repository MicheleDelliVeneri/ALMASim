import { render, screen, within } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';

import type { Run } from '#lib/api/v2/client.js';
import RunsTable from './RunsTable.svelte';

const run = (overrides: Partial<Run>): Run => ({
	id: '01JRUN0000000000000000000A',
	short_id: '000000000A',
	owner: 'astro',
	command: 'image image-from-ms',
	argv: ['image', 'image-from-ms', 'p.csv', 'out'],
	title: '',
	executor: 'slurm',
	resources: { partition: 'zwickyA-CPU', account: null, cpus: 4, memory: '16G', time: '1:00:00' },
	params: {},
	parents: [],
	slurm_job_id: '6402',
	state: 'running',
	finished: false,
	reason: '',
	created_at: 1000,
	started_at: 1000,
	finished_at: null,
	host: null,
	exit_code: null,
	signal: null,
	run_dir: '/runs/x',
	...overrides
});

describe('RunsTable', () => {
	const runs = [
		run({}),
		run({
			id: '01JRUN0000000000000000000B',
			short_id: '000000000B',
			owner: 'teammate',
			command: 'metadata',
			title: 'Band 3 galaxies',
			state: 'queued',
			started_at: null,
			reason: 'Priority',
			executor: 'server',
			slurm_job_id: null
		})
	];

	it('lists each run with its state, where and how long, linking to its page', () => {
		render(RunsTable, { runs, now: 1000 + 3 * 3600 + 12 * 60, showOwner: true });
		const table = screen.getByRole('table');
		const rows = within(table).getAllByRole('row');
		expect(rows).toHaveLength(3);
		expect(within(rows[1]).getByText('Running')).toBeInTheDocument();
		expect(within(rows[1]).getByText('Slurm 6402 · zwickyA-CPU')).toBeInTheDocument();
		expect(within(rows[1]).getByText('3 h 12 m')).toBeInTheDocument();
		expect(within(rows[1]).getByRole('link', { name: 'image image-from-ms' })).toHaveAttribute(
			'href',
			'/jobs/01JRUN0000000000000000000A'
		);
		// A title replaces the command; why it waits is shown.
		expect(within(rows[2]).getByRole('link', { name: 'Band 3 galaxies' })).toBeInTheDocument();
		expect(within(rows[2]).getByText('Priority')).toBeInTheDocument();
		expect(within(rows[2]).getByText('teammate')).toBeInTheDocument();
		expect(within(rows[2]).getByText('console server')).toBeInTheDocument();
	});

	it('leaves the owner column out of your own list', () => {
		render(RunsTable, { runs, now: 2000 });
		const headers = within(screen.getByRole('table')).getAllByRole('columnheader');
		expect(headers.map((h) => h.textContent?.trim())).not.toContain('Who');
	});
});
