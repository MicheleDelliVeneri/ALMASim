<!-- Runs as a table on wide screens and as cards on phones; each links to its page. -->
<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Run } from '#lib/api/v2/client.js';
	import { elapsedLabel, whereLabel } from '#lib/utils/jobs.js';
	import StateChip from './StateChip.svelte';

	interface Props {
		runs: Run[];
		/** Seconds since the epoch, for the elapsed times. */
		now: number;
		/** Show whose run it is (the team view). */
		showOwner?: boolean;
	}

	let { runs, now, showOwner = false }: Props = $props();

	const started = (run: Run) =>
		new Date(run.created_at * 1000).toLocaleString(undefined, {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
</script>

<div class="hidden overflow-x-auto md:block">
	<table class="w-full border-collapse text-sm">
		<thead>
			<tr class="text-left text-xs font-semibold tracking-wide text-fg-muted uppercase">
				<th scope="col" class="border-b border-line-strong px-3 py-2">State</th>
				<th scope="col" class="border-b border-line-strong px-3 py-2">Command</th>
				{#if showOwner}<th scope="col" class="border-b border-line-strong px-3 py-2">Who</th>{/if}
				<th scope="col" class="border-b border-line-strong px-3 py-2">Where</th>
				<th scope="col" class="border-b border-line-strong px-3 py-2">Submitted</th>
				<th scope="col" class="border-b border-line-strong px-3 py-2">Elapsed</th>
			</tr>
		</thead>
		<tbody>
			{#each runs as run (run.id)}
				<tr class="border-b border-sunk align-middle hover:bg-canvas">
					<td class="px-3 py-2.5"><StateChip state={run.state} /></td>
					<td class="max-w-md px-3 py-2.5">
						<a
							href={resolve('/jobs/[id]', { id: run.id })}
							class="font-semibold text-navy no-underline hover:underline"
							>{run.title || run.command}</a
						>
						<div class="truncate font-mono text-xs text-fg-dim" title={run.argv.join(' ')}>
							{run.title ? `${run.command} · ` : ''}{run.short_id}
						</div>
						{#if run.reason && !run.finished}
							<div class="text-xs text-fg-muted">{run.reason}</div>
						{/if}
					</td>
					{#if showOwner}<td class="px-3 py-2.5 text-fg-muted">{run.owner}</td>{/if}
					<td class="px-3 py-2.5 text-fg-muted">{whereLabel(run)}</td>
					<td class="px-3 py-2.5 whitespace-nowrap text-fg-muted">{started(run)}</td>
					<td class="px-3 py-2.5 font-mono text-xs whitespace-nowrap text-fg-muted"
						>{elapsedLabel(run, now)}</td
					>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<ul class="flex flex-col gap-2.5 md:hidden">
	{#each runs as run (run.id)}
		<li>
			<a
				href={resolve('/jobs/[id]', { id: run.id })}
				class="flex flex-col gap-2 rounded-lg border border-line bg-surface p-3.5 text-fg no-underline"
			>
				<span class="flex items-center gap-2">
					<StateChip state={run.state} />
					{#if showOwner}<span class="text-xs text-fg-muted">{run.owner}</span>{/if}
					<span class="ml-auto font-mono text-xs text-fg-muted">{elapsedLabel(run, now)}</span>
				</span>
				<span class="text-base font-semibold text-navy">{run.title || run.command}</span>
				<span class="text-xs text-fg-muted">{run.reason || whereLabel(run)}</span>
			</a>
		</li>
	{/each}
</ul>
