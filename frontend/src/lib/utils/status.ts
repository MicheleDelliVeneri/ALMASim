/**
 * Job statuses shared by downloads and simulations: which ones are still
 * active, which are final, and the badge tone each one shows.
 */

const ACTIVE = new Set(['queued', 'pending', 'starting', 'running', 'downloading']);
const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

export type StatusTone = 'active' | 'queued' | 'ok' | 'danger' | 'neutral';

/** The job is waiting or working: poll it, offer to cancel it. */
export function isActiveStatus(status: string): boolean {
	return ACTIVE.has(status);
}

/** The job has finished, one way or another. */
export function isTerminalStatus(status: string): boolean {
	return TERMINAL.has(status);
}

export function statusTone(status: string): StatusTone {
	if (status === 'queued') return 'queued';
	if (ACTIVE.has(status)) return 'active';
	if (status === 'completed') return 'ok';
	if (status === 'failed') return 'danger';
	return 'neutral';
}

export const STATUS_TONE_CLASSES: Record<StatusTone, string> = {
	active: 'bg-accent-soft text-accent',
	queued: 'bg-yellow-100 text-yellow-800',
	ok: 'bg-ok-soft text-ok',
	danger: 'bg-danger-soft text-danger',
	neutral: 'bg-surface text-fg-muted'
};
