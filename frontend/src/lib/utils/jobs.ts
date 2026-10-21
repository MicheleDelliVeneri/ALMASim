/**
 * Runs as the jobs pages show them: state labels and tones, durations, the
 * command line to copy, where a run ran, and the list filters.
 */
import type { JobState, Run } from '#lib/api/v2/client.js';

export const STATE_LABEL: Record<JobState, string> = {
	pending: 'Waiting',
	queued: 'Queued',
	running: 'Running',
	succeeded: 'Succeeded',
	failed: 'Failed',
	cancelled: 'Cancelled',
	lost: 'Lost'
};

/** Chip classes: background, text and dot, from the semantic tokens. */
export const STATE_CLASSES: Record<JobState, { chip: string; dot: string }> = {
	pending: { chip: 'bg-sunk text-fg-muted', dot: 'bg-fg-dim' },
	queued: { chip: 'bg-sunk text-fg-muted', dot: 'bg-fg-dim' },
	running: { chip: 'bg-info-soft text-info', dot: 'bg-info' },
	succeeded: { chip: 'bg-ok-soft text-ok', dot: 'bg-ok' },
	failed: { chip: 'bg-danger-soft text-danger', dot: 'bg-danger' },
	cancelled: { chip: 'bg-warn-soft text-warn', dot: 'bg-warn' },
	lost: { chip: 'bg-danger-soft text-danger', dot: 'bg-danger' }
};

/** "45 s", "12 m", "3 h 12 m", "2 d 4 h". */
export function formatDuration(seconds: number): string {
	const s = Math.max(0, Math.floor(seconds));
	if (s < 60) return `${s} s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m} m`;
	const h = Math.floor(m / 60);
	if (h < 24) return `${h} h ${m % 60} m`;
	return `${Math.floor(h / 24)} d ${h % 24} h`;
}

/** How long the run has been running (or ran), in seconds; null before it started. */
export function runSeconds(run: Run, now: number): number | null {
	if (run.started_at == null) return null;
	return (run.finished_at ?? now) - run.started_at;
}

/** "3 h 12 m" running, "waiting 5 m" before it starts, "—" when it never started. */
export function elapsedLabel(run: Run, now: number): string {
	const ran = runSeconds(run, now);
	if (ran != null) return formatDuration(ran);
	if (run.finished) return '—';
	return `waiting ${formatDuration(now - run.created_at)}`;
}

const SAFE = /^[A-Za-z0-9_\-+=.,/:@%]+$/;

/** One argument as a POSIX shell reads it back. */
export function shellQuote(arg: string): string {
	if (SAFE.test(arg)) return arg;
	return `'${arg.replace(/'/g, `'\\''`)}'`;
}

/** The command line to paste in a terminal. */
export function commandLine(argv: string[]): string {
	return ['almasim', ...argv].map(shellQuote).join(' ');
}

/** Where the run ran: its Slurm job and partition, or the console's own machine. */
export function whereLabel(run: Run): string {
	if (run.executor !== 'slurm') return 'console server';
	const id = run.slurm_job_id ? `Slurm ${run.slurm_job_id}` : 'Slurm';
	return run.resources.partition ? `${id} · ${run.resources.partition}` : id;
}

export type OwnerFilter = 'mine' | 'team';
export type StateFilter = 'all' | 'active' | 'failed' | 'succeeded';

export const STATE_FILTERS: Record<StateFilter, (run: Run) => boolean> = {
	all: () => true,
	active: (run) => !run.finished,
	failed: (run) => run.state === 'failed' || run.state === 'lost',
	succeeded: (run) => run.state === 'succeeded'
};

export interface RunFilter {
	owner: OwnerFilter;
	/** Who the console runs as; null until the first snapshot says. */
	user: string | null;
	state: StateFilter;
	query: string;
}

/** The runs a filter keeps, in the order given. */
export function filterRuns(runs: Run[], filter: RunFilter): Run[] {
	const words = filter.query.toLowerCase().split(/\s+/).filter(Boolean);
	return runs.filter((run) => {
		if (filter.owner === 'mine' && filter.user !== null && run.owner !== filter.user) return false;
		if (!STATE_FILTERS[filter.state](run)) return false;
		if (words.length === 0) return true;
		const text = [run.command, run.title, run.id, run.owner, run.slurm_job_id ?? '', ...run.argv]
			.join(' ')
			.toLowerCase();
		return words.every((word) => text.includes(word));
	});
}
