<script lang="ts">
	import type { QueryJob } from './queryJob.js';

	interface Props {
		jobs: QueryJob[];
		selectedJobId: string | null;
		onShowAll: () => void;
		onStop: (id: string) => void;
		onLoad: (id: string) => void;
		onSave: (id: string) => void;
		onSavePreset: (id: string) => void;
		onDismiss: (id: string) => void;
	}

	let { jobs, selectedJobId, onShowAll, onStop, onLoad, onSave, onSavePreset, onDismiss }: Props =
		$props();
</script>

{#if jobs.length > 0}
	<div class="space-y-2">
		{#if selectedJobId}
			<div
				class="flex items-center justify-between rounded-md border border-accent-line bg-accent-soft px-4 py-2 text-xs text-accent"
			>
				<span
					>Showing results for query <span class="font-mono">{selectedJobId.slice(0, 8)}</span
					></span
				>
				<button onclick={onShowAll} class="underline hover:text-accent">Show all</button>
			</div>
		{/if}
		{#each jobs as job (job.id)}
			<div
				class="flex items-center justify-between rounded-md border px-4 py-2 text-sm
				{selectedJobId === job.id ? 'ring-2 ring-indigo-400' : ''}
				{job.status === 'error'
					? 'border-danger-line bg-danger-soft text-danger'
					: job.status === 'cancelled'
						? 'border-line bg-surface text-fg-dim'
						: job.status === 'done'
							? 'border-ok-line bg-ok-soft text-ok'
							: 'border-accent-line bg-accent-soft text-accent'}"
			>
				<div class="flex items-center gap-2 min-w-0">
					{#if job.status === 'fetching' || job.status === 'starting'}
						<svg class="h-4 w-4 shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
							<circle
								class="opacity-25"
								cx="12"
								cy="12"
								r="10"
								stroke="currentColor"
								stroke-width="4"
							/>
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
						</svg>
					{/if}
					<span class="truncate">
						{job.status === 'error' ? job.error : job.message}
						<span class="ml-1 font-mono text-xs opacity-60">{job.id.slice(0, 8)}</span>
					</span>
				</div>
				<div class="ml-4 flex shrink-0 gap-2">
					{#if job.status === 'fetching'}
						<button
							onclick={() => onStop(job.id)}
							class="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700"
							>Stop</button
						>
					{:else}
						{#if job.status === 'done' && job.rows.length > 0}
							<button
								onclick={() => onLoad(job.id)}
								class="rounded-md px-2 py-1 text-xs font-medium ring-1
									{selectedJobId === job.id
									? 'bg-primary text-white ring-indigo-600 hover:bg-primary-hover'
									: 'bg-surface text-accent ring-indigo-300 hover:bg-accent-soft'}"
								>{selectedJobId === job.id ? 'Loaded' : 'Load'}</button
							>
							<button
								onclick={() => onSave(job.id)}
								class="rounded-md bg-surface px-2 py-1 text-xs font-medium text-fg-muted ring-1 ring-gray-300 hover:bg-hover"
								>Save</button
							>
							<button
								onclick={() => onSavePreset(job.id)}
								class="rounded-md bg-surface px-2 py-1 text-xs font-medium text-accent ring-1 ring-indigo-300 hover:bg-accent-soft"
								>Save as Preset</button
							>
						{/if}
						<button
							onclick={() => onDismiss(job.id)}
							class="rounded-md bg-surface px-2 py-1 text-xs font-medium text-fg-muted ring-1 ring-gray-300 hover:bg-hover"
							>Dismiss</button
						>
					{/if}
				</div>
			</div>
		{/each}
	</div>
{/if}
