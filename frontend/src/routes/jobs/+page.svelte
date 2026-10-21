<script lang="ts">
	import { clock } from '#lib/clock.svelte.js';
	import RunsTable from '#lib/components/jobs/RunsTable.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { filterRuns, type OwnerFilter, type StateFilter } from '#lib/utils/jobs.js';

	// How many runs the console's event stream follows (EventHub's limit).
	const FOLLOWED = 200;

	let owner = $state<OwnerFilter>('mine');
	let stateFilter = $state<StateFilter>('all');
	let query = $state('');

	$effect(() => clock.use());

	const OWNERS: { value: OwnerFilter; label: string }[] = [
		{ value: 'mine', label: 'Mine' },
		{ value: 'team', label: 'Team' }
	];
	const STATES: { value: StateFilter; label: string }[] = [
		{ value: 'all', label: 'All' },
		{ value: 'active', label: 'Active' },
		{ value: 'failed', label: 'Failed' },
		{ value: 'succeeded', label: 'Succeeded' }
	];

	const shown = $derived(
		filterRuns(jobs.list, { owner, user: jobs.user, state: stateFilter, query })
	);
	const counts = $derived(
		Object.fromEntries(
			STATES.map(({ value }) => [
				value,
				filterRuns(jobs.list, { owner, user: jobs.user, state: value, query: '' }).length
			])
		) as Record<StateFilter, number>
	);
</script>

<svelte:head><title>Jobs · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-wrap items-end gap-4">
		<div class="flex min-w-0 flex-1 flex-col gap-1">
			<p class="alma-kicker">Runs</p>
			<h1 class="m-0 text-3xl">Jobs</h1>
		</div>
		<p class="text-sm text-fg-muted" aria-live="polite">
			{#if jobs.connected}
				{jobs.active.length} active · live
			{:else if jobs.loaded}
				Reconnecting…
			{/if}
		</p>
	</div>

	<div class="flex flex-wrap items-center gap-3">
		<div role="group" aria-label="Whose jobs" class="flex gap-1 rounded-full bg-sunk p-1">
			{#each OWNERS as option (option.value)}
				<button
					type="button"
					class="rounded-full px-4 py-1.5 text-sm font-medium text-fg-muted aria-pressed:bg-navy aria-pressed:text-white"
					aria-pressed={owner === option.value}
					onclick={() => (owner = option.value)}>{option.label}</button
				>
			{/each}
		</div>
		<div role="group" aria-label="Which states" class="flex flex-wrap gap-2">
			{#each STATES as option (option.value)}
				<button
					type="button"
					class="rounded-full border border-line-strong bg-surface px-3.5 py-1.5 text-sm font-medium text-navy hover:bg-hover aria-pressed:border-navy aria-pressed:bg-navy aria-pressed:text-white"
					aria-pressed={stateFilter === option.value}
					onclick={() => (stateFilter = option.value)}
					>{option.label}
					<span class="ml-1 text-xs opacity-75">{counts[option.value]}</span></button
				>
			{/each}
		</div>
		<label class="ml-auto flex min-w-56 flex-1 sm:flex-none">
			<span class="sr-only">Search jobs</span>
			<input
				type="search"
				class="w-full rounded-md border px-3 py-2 text-sm"
				placeholder="Search command, id, path…"
				bind:value={query}
			/>
		</label>
	</div>

	<section class="alma-panel rounded-lg p-0 md:p-2" aria-label="Job list">
		{#if !jobs.loaded}
			<p class="flex items-center gap-2 p-6 text-sm text-fg-muted">
				<Spinner /> Loading jobs…
			</p>
		{:else if shown.length === 0}
			<p class="p-6 text-sm text-fg-muted">
				{#if jobs.list.length === 0}
					No jobs yet. Runs started from this console, or with <code>almasim submit</code>, appear
					here.
				{:else}
					No {owner === 'mine' ? 'jobs of yours' : 'jobs'} match these filters.
				{/if}
			</p>
		{:else}
			<RunsTable runs={shown} now={clock.now} showOwner={owner === 'team'} />
		{/if}
	</section>
	{#if jobs.list.length >= FOLLOWED}
		<p class="text-xs text-fg-dim">
			The console follows the latest {FOLLOWED} runs; older ones are still in the runs folder and in
			<code>almasim jobs list</code>.
		</p>
	{/if}
</div>
