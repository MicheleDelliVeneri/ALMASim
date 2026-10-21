<!--
	Simulate ALMA observations: pick rows of a metadata CSV (from the Archive
	page, or any saved one), choose the sky model and options, see how big
	the cubes will be, and submit one run per row. ?metadata=<csv path>
	opens a CSV, &rows=0,3 ticks rows in it.
-->
<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Run } from '#lib/api/v2/client.js';
	import type { components } from '#lib/api/v2/schema.js';
	import ResultsTable from '#lib/components/archive/ResultsTable.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import CommandSections from '#lib/components/run/CommandSections.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { toRecords } from '#lib/utils/archive.js';
	import type { Command, Values } from '#lib/utils/commandForm.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import { ROW_FIELDS, batchGb, simulationArgv, simulationDefaults } from '#lib/utils/simulate.js';

	type Estimate = components['schemas']['Estimate'];

	let simulation = $state<Command | null>(null);
	let values = $state<Values>({});
	let loadError = $state<unknown>(null);
	let metadataPath = $state(page.url.searchParams.get('metadata') ?? '');
	let table = $state<{ columns: string[]; records: Record<string, string>[] } | null>(null);
	let tableError = $state<unknown>(null);
	let selected = $state(new Set<number>());
	let estimate = $state<Estimate | null>(null);
	let estimateError = $state<unknown>(null);
	let estimating = $state(false);
	let submitting = $state(false);
	let submitted = $state<string[]>([]);

	const rows = $derived([...selected].sort((a, b) => a - b));
	const perRow = $derived(Number(values.simulations_per_row) || 1);

	$effect(() => {
		void (async () => {
			try {
				const { data } = await api.GET('/api/v2/commands');
				simulation = data?.find((c) => c.name === 'simulation') ?? null;
				if (simulation) values = simulationDefaults(simulation);
			} catch (err) {
				loadError = err;
			}
		})();
	});

	// The metadata rows, once a CSV is chosen.
	$effect(() => {
		const path = metadataPath.trim();
		table = null;
		tableError = null;
		if (!path.endsWith('.csv')) return;
		let stale = false;
		api
			.GET('/api/v2/files/csv', { params: { query: { path, start: 0, rows: 5000 } } })
			.then(({ data }) => {
				if (stale || !data) return;
				table = { columns: data.columns, records: toRecords(data.columns, data.rows) };
				const wanted = (page.url.searchParams.get('rows') ?? '')
					.split(',')
					.map(Number)
					.filter((i) => Number.isInteger(i) && i >= 0 && i < data.rows.length);
				selected = new Set(wanted.length ? wanted : data.rows.length === 1 ? [0] : []);
			})
			.catch((err) => !stale && (tableError = err));
		return () => (stale = true);
	});

	// How big the cubes will be: for the first ticked row, a moment after a change.
	$effect(() => {
		const body = {
			metadata_path: metadataPath.trim(),
			row_idx: rows[0] ?? -1,
			n_pix: Number(values.n_pix) || null,
			n_channels: Number(values.n_channels) || null,
			source_type: String(values.source_type || 'point'),
			instrument_geometry: String(values.instrument_geometry || 'legacy')
		};
		if (!table || body.row_idx < 0) {
			estimate = null;
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			estimating = true;
			try {
				const { data } = await api.POST('/api/v2/simulation/estimate', {
					body,
					signal: controller.signal
				});
				estimate = data ?? null;
				estimateError = null;
			} catch (err) {
				if (!controller.signal.aborted) {
					estimate = null;
					estimateError = err;
				}
			} finally {
				if (!controller.signal.aborted) estimating = false;
			}
		}, 500);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	async function submit() {
		if (!simulation || !rows.length) return;
		submitting = true;
		const made: Run[] = [];
		try {
			for (const i of rows) {
				const source = table?.records[i]?.ALMA_source_name || `row ${i}`;
				const { data } = await api.POST('/api/v2/jobs', {
					body: {
						argv: simulationArgv(simulation, values, metadataPath.trim(), i, rows.length > 1),
						params: { ...values, metadata_path: metadataPath.trim(), row_idx: i },
						title: `Simulation · ${source}`
					}
				});
				if (data) {
					jobs.put(data);
					made.push(data);
				}
			}
			toasts.show(`Submitted ${made.length} simulation${made.length === 1 ? '' : 's'}`, 'ok');
		} catch (err) {
			toasts.show(
				`Submitted ${made.length} of ${rows.length}, then: ${errorMessage(err)}`,
				'danger'
			);
		} finally {
			submitted = [...made.map((r) => r.id), ...submitted];
			submitting = false;
		}
	}

	const gb = (n: number) => (n >= 100 ? n.toFixed(0) : n >= 1 ? n.toFixed(1) : n.toFixed(3));
</script>

<svelte:head><title>Simulate · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Workflow</p>
		<h1 class="m-0 text-3xl">Simulate observations</h1>
		<p class="m-0 text-fg-muted">
			Pick observations from a metadata CSV, choose a sky model, and simulate what ALMA would see:
			one Slurm job per observation.
		</p>
	</div>

	{#if loadError}<ErrorMessage error={loadError} title="Something did not load" />{/if}

	<section class="alma-panel flex flex-col gap-4 rounded-lg p-5" aria-labelledby="m-h">
		<h2 id="m-h" class="m-0 text-xl font-semibold">Observations</h2>
		<PathPicker
			bind:value={metadataPath}
			label="Metadata CSV"
			kind="file"
			hint="The results of an archive search (Archive page), or any CSV from almasim metadata."
		/>
		{#if tableError}
			<ErrorMessage error={tableError} title="Could not read the CSV" />
		{:else if metadataPath.trim().endsWith('.csv') && !table}
			<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Reading…</p>
		{:else if table}
			<ResultsTable records={table.records} columns={table.columns} bind:selected />
		{/if}
	</section>

	{#if simulation && table}
		<p class="m-0 text-sm text-fg-muted">
			<code>{'{run_dir}'}</code> in a path is each run's own folder (its job page lists what is in
			it). By default every simulation keeps its FITS products and its ML shard there; a folder you
			choose instead gets one <code>row-&lt;n&gt;</code> folder per observation.
		</p>
		<CommandSections command={simulation} bind:values hide={ROW_FIELDS} />

		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="e-h">
			<div class="flex items-center gap-2">
				<h2 id="e-h" class="m-0 text-xl font-semibold">Size</h2>
				{#if estimating}<Spinner />{/if}
			</div>
			{#if !rows.length}
				<p class="m-0 text-sm text-fg-muted">Tick at least one observation.</p>
			{:else if estimateError}
				<ErrorMessage error={estimateError} title="Could not estimate it" />
			{:else if estimate}
				<p class="m-0 text-sm">
					Each simulation of <strong>{estimate.source_name}</strong> (band {estimate.band}) makes a
					cube of {estimate.cube_shape.join(' × ')} (channels × pixels²), {estimate.cell_size_arcsec.toFixed(
						3
					)}″ pixels for a {estimate.beam_size_arcsec.toFixed(2)}″ beam: about
					<strong>{gb(estimate.estimated_standard_output_gb)} GB</strong> of standard outputs.
				</p>
				<p class="m-0 text-sm">
					{rows.length} observation{rows.length === 1 ? '' : 's'} × {perRow} per observation: about
					<strong
						>{gb(batchGb(estimate.estimated_standard_output_gb, rows.length, perRow))} GB</strong
					>
					in all (if the others are like the first).
				</p>
				<p class="m-0 text-xs text-fg-dim">{estimate.note}</p>
			{/if}
			<div>
				<button
					type="button"
					class="btn-primary bg-magenta hover:bg-accent"
					disabled={!rows.length || submitting}
					onclick={submit}
					>{submitting
						? 'Submitting…'
						: `Simulate ${rows.length} observation${rows.length === 1 ? '' : 's'}`}</button
				>
			</div>
		</section>
	{/if}

	{#if submitted.length}
		<section class="alma-panel flex flex-col gap-2 rounded-lg p-5" aria-labelledby="r-h">
			<h2 id="r-h" class="m-0 text-xl font-semibold">Submitted</h2>
			<ul class="m-0 flex list-none flex-col gap-1.5 p-0">
				{#each submitted as id (id)}
					{@const run = jobs.get(id)}
					<li class="flex items-center gap-3 text-sm">
						{#if run}<StateChip state={run.state} />{/if}
						<a href={resolve('/jobs/[id]', { id })}>{run?.title ?? id}</a>
					</li>
				{/each}
			</ul>
		</section>
	{/if}
</div>
