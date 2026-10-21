<!--
	Home: your jobs, what needs your attention, the partitions' load, the
	team's latest runs and your latest results. Live from the jobs store;
	the cluster from /api/v2/cluster (cached 30 s on the console).
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import { api } from '#lib/api/v2/client.js';
	import type { components } from '#lib/api/v2/schema.js';
	import { clock } from '#lib/clock.svelte.js';
	import RunsTable from '#lib/components/jobs/RunsTable.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { homeSummary } from '#lib/utils/home.js';

	type Cluster = components['schemas']['Cluster'];

	let cluster = $state<Cluster | null>(null);

	$effect(() => clock.use());

	$effect(() => {
		const load = () =>
			api
				.GET('/api/v2/cluster')
				.then(({ data }) => (cluster = data ?? null))
				.catch(() => (cluster = null));
		void load();
		const timer = setInterval(load, 60_000);
		return () => clearInterval(timer);
	});

	const summary = $derived(homeSummary(jobs.list, jobs.user, clock.now));
	const today = new Date().toLocaleDateString(undefined, {
		weekday: 'long',
		day: 'numeric',
		month: 'long'
	});
	const productsOf = (dir: string) => `${resolve('/products')}?dir=${encodeURIComponent(dir)}`;
</script>

<svelte:head><title>ALMASim</title></svelte:head>

<div class="flex flex-col gap-6">
	<div class="flex flex-wrap items-end gap-4">
		<div class="flex min-w-0 flex-1 basis-96 flex-col gap-1">
			<p class="alma-kicker">{today}</p>
			<h1 class="m-0 text-4xl leading-tight">{summary.headline}</h1>
		</div>
		<div class="flex flex-wrap gap-2">
			<a class="btn-secondary no-underline" href={resolve('/archive')}>Search the archive</a>
			<a class="btn-primary bg-magenta no-underline hover:bg-accent" href={resolve('/run')}
				>New run</a
			>
		</div>
	</div>

	<section class="alma-panel overflow-hidden rounded-lg" aria-labelledby="h-mine">
		<div class="flex flex-wrap items-center gap-3 border-b border-line px-5 py-4">
			<h2 id="h-mine" class="m-0 text-xl font-semibold">Your jobs</h2>
			<span class="text-sm text-fg-muted">{jobs.connected ? 'live' : 'offline'}</span>
			<a class="ml-auto text-sm font-medium" href={resolve('/jobs')}>All jobs</a>
		</div>
		{#if !jobs.loaded}
			<p class="flex items-center gap-2 p-5 text-sm text-fg-muted"><Spinner /> Loading…</p>
		{:else if summary.mine.length}
			<div class="p-0 md:p-2"><RunsTable runs={summary.mine} now={clock.now} /></div>
		{:else}
			<p class="p-5 text-sm text-fg-muted">
				No jobs yet. <a href={resolve('/run')}>Run a command</a>, or start from the
				<a href={resolve('/archive')}>archive</a>.
			</p>
		{/if}
	</section>

	<div class="grid gap-6 md:grid-cols-2">
		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="h-parts">
			<h2 id="h-parts" class="m-0 text-xl font-semibold">Partitions</h2>
			{#if !cluster}
				<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Asking Slurm…</p>
			{:else if !cluster.available}
				<p class="m-0 text-sm text-fg-muted">
					No Slurm here: runs go to the console's own machine.
				</p>
			{:else}
				{#each cluster.partitions.filter((p) => p.up) as p (p.name)}
					<div class="grid grid-cols-[8rem_1fr_3rem] items-center gap-3">
						<span class="truncate font-mono text-sm">{p.name}</span>
						<div
							class="h-2.5 overflow-hidden rounded-full bg-sunk"
							role="meter"
							aria-label="{p.name} busy"
							aria-valuemin="0"
							aria-valuemax="100"
							aria-valuenow={Math.round((p.load ?? 0) * 100)}
						>
							<div class="h-full bg-navy" style="width: {Math.round((p.load ?? 0) * 100)}%"></div>
						</div>
						<span class="text-right text-sm text-fg-muted">{Math.round((p.load ?? 0) * 100)}%</span>
					</div>
				{/each}
				<p class="m-0 text-sm text-fg-muted">
					Your queue: {cluster.queue.filter((j) => j.state === 'RUNNING').length} running,
					{cluster.queue.filter((j) => j.state === 'PENDING').length} pending.
					{#if cluster.defaults.partition}Default partition {cluster.defaults.partition}.{/if}
				</p>
			{/if}
		</section>

		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="h-team">
			<div class="flex items-center gap-2">
				<h2 id="h-team" class="m-0 text-xl font-semibold">Team runs</h2>
				<span class="rounded-full bg-sunk px-2 py-0.5 text-xs text-fg-muted">read only</span>
			</div>
			{#each summary.team as run (run.id)}
				<a
					href={resolve('/jobs/[id]', { id: run.id })}
					class="flex items-baseline gap-3 border-b border-sunk pb-2 text-fg no-underline"
				>
					<span class="w-28 shrink-0 truncate text-sm font-semibold text-fg-muted">{run.owner}</span
					>
					<span class="min-w-0 flex-1 truncate text-sm">{run.title || run.command}</span>
					<StateChip state={run.state} />
				</a>
			{:else}
				<p class="m-0 text-sm text-fg-muted">Nobody else has run anything lately.</p>
			{/each}
		</section>
	</div>

	{#if summary.results.length}
		<section class="flex flex-col gap-3" aria-labelledby="h-results">
			<h2 id="h-results" class="m-0 text-xl font-semibold">Recent results</h2>
			<div class="grid grid-cols-[repeat(auto-fit,minmax(15rem,1fr))] gap-4">
				{#each summary.results as run (run.id)}
					<!-- eslint-disable svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended -->
					<a
						href={productsOf(run.run_dir)}
						class="alma-panel flex flex-col gap-1.5 rounded-lg p-4 text-fg no-underline hover:border-line-strong"
					>
						<span class="font-semibold text-navy">{run.title || run.command}</span>
						<span class="font-mono text-xs text-fg-muted">{run.command} · {run.short_id}</span>
						<span class="text-xs text-fg-dim"
							>{new Date((run.finished_at ?? run.created_at) * 1000).toLocaleString()}</span
						>
					</a>
					<!-- eslint-enable svelte/no-navigation-without-resolve -->
				{/each}
			</div>
		</section>
	{/if}
</div>
