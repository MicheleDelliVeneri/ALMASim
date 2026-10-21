/** What the home page says: a headline, your jobs, the team's, and your latest results. */
import type { Run } from '#lib/api/v2/client.js';

const DAY = 86_400;

export interface HomeSummary {
	headline: string;
	/** Your active runs, then those that need attention, then your latest finished; six at most. */
	mine: Run[];
	/** Everyone else's latest runs. */
	team: Run[];
	/** Your latest successes, for their outputs. */
	results: Run[];
	/** Your runs that failed or were lost in the last day. */
	attention: Run[];
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

/** `runs` newest first; `user` who the console runs as (null before it says). */
export function homeSummary(runs: Run[], user: string | null, now: number): HomeSummary {
	const isMine = (run: Run) => user === null || run.owner === user;
	const mine = runs.filter(isMine);
	const active = mine.filter((r) => !r.finished);
	const attention = mine.filter(
		(r) =>
			(r.state === 'failed' || r.state === 'lost') && now - (r.finished_at ?? r.created_at) < DAY
	);
	const running = active.filter((r) => r.state === 'running').length;
	const waiting = active.length - running;

	let headline = 'Nothing running';
	if (active.length) {
		headline = running
			? `${plural(running, 'job')} running${waiting ? `, ${waiting} waiting` : ''}`
			: `${plural(waiting, 'job')} waiting`;
	}
	if (attention.length)
		headline += `, ${attention.length} need${attention.length === 1 ? 's' : ''} your attention`;

	return {
		headline,
		mine: [
			...active,
			...attention,
			...mine.filter((r) => r.finished && !attention.includes(r))
		].slice(0, 6),
		team: runs.filter((r) => !isMine(r)).slice(0, 5),
		results: mine.filter((r) => r.state === 'succeeded').slice(0, 4),
		attention
	};
}
