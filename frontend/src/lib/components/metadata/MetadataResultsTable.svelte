<script lang="ts">
	import type { MetadataResponse } from '$lib/api/metadata';

	interface Props {
		results: MetadataResponse | null;
		loading: boolean;
		fetching: boolean;
		onClear: () => void;
		onLoad: () => void;
		onSave: () => void;
		saving: boolean;
	}

	let { results, loading, fetching, onClear, onLoad, onSave, saving }: Props = $props();

	let isCollapsed = $state(false);
	let columnsMenuOpen = $state(false);

	// All known columns with their display labels (kept in preferred display order)
	const columnLabels: Record<string, string> = {
		ALMA_source_name: 'Source Name',
		Band: 'Band',
		Array_type: 'Array Type',
		antenna_arrays: 'Antenna Arrays',
		'Ang.res.': 'Ang. Res. (arcsec)',
		'Obs.date': 'Obs. Date',
		Project_abstract: 'Project Abstract',
		science_keyword: 'Science Keyword',
		scientific_category: 'Science Category',
		QA2_status: 'QA2 Status',
		Type: 'Type',
		PWV: 'PWV',
		SB_name: 'SB Name',
		'Vel.res.': 'Vel. Res. (km/s)',
		RA: 'R.A. (deg)',
		Dec: 'Dec (deg)',
		FOV: 'FOV (arcsec)',
		'Int.Time': 'Int. Time (s)',
		Cont_sens_mJybeam: 'Cont. Sens. (mJy/beam)',
		Line_sens_10kms_mJybeam: 'Line Sens. 10km/s (mJy/beam)',
		Bandwidth: 'Bandwidth (GHz)',
		Freq: 'Frequency (GHz)',
		'Freq.sup.': 'Freq. Support',
		proposal_id: 'Proposal ID',
		member_ous_uid: 'Member OUS UID',
		group_ous_uid: 'Group OUS UID',
	};

	// Preferred display order (first columns shown in table)
	const preferredOrder = [
		'ALMA_source_name',
		'Band',
		'Array_type',
		'Ang.res.',
		'Obs.date',
		'Project_abstract',
		'science_keyword',
		'scientific_category',
		'QA2_status',
		'Type',
	];

	// All columns available for the Columns menu — always the full ALMA set, in display order
	const menuColumns = [
		...preferredOrder,
		...Object.keys(columnLabels).filter((c) => !preferredOrder.includes(c)),
	];

	// Columns hidden by default (verbose / less commonly needed)
	const defaultHidden = new Set([
		'antenna_arrays',
		'PWV',
		'SB_name',
		'Vel.res.',
		'Cont_sens_mJybeam',
		'Line_sens_10kms_mJybeam',
		'Freq.sup.',
		'proposal_id',
		'group_ous_uid',
	]);
	let hiddenColumns = $state(new Set<string>(defaultHidden));

	// Client-side column filters (reset on new results)
	let columnFilters = $state<Record<string, string>>({});

	$effect(() => {
		if (results) columnFilters = {};
	});

	const hasActiveFilters = $derived(Object.values(columnFilters).some((v) => v.length > 0));

	// Columns that actually appear in the current result data
	const dataColumnSet = $derived.by(() => {
		const data = results?.data;
		if (!data || data.length === 0) return new Set<string>();
		return new Set(Object.keys(data[0]));
	});

	// Visible table columns: from menuColumns, not hidden, and present in data
	const tableColumns = $derived(
		menuColumns.filter((c) => !hiddenColumns.has(c) && dataColumnSet.has(c))
	);

	// Apply column filters to data
	const filteredData = $derived.by(() => {
		const data = results?.data ?? [];
		const active = Object.entries(columnFilters).filter(([, v]) => v.length > 0);
		if (active.length === 0) return data;
		return data.filter((row) =>
			active.every(([col, val]) =>
				String(row[col] ?? '')
					.toLowerCase()
					.includes(val.toLowerCase())
			)
		);
	});

	// Virtual scrolling constants
	const ROW_HEIGHT = 36;        // px — matches row height below
	const OVERSCAN = 15;          // extra rows above/below viewport
	const CONTAINER_HEIGHT = 480; // must match max-height in CSS

	// Mirror-scroll: top scrollbar synced to the real scroll container
	let scrollContainer = $state<HTMLDivElement | null>(null);
	let mirrorBar = $state<HTMLDivElement | null>(null);
	let tableScrollWidth = $state(0);
	let scrollTop = $state(0);

	// Virtual window derived from scroll position
	const totalRows = $derived(filteredData.length);
	const virtualStart = $derived(Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN));
	const virtualEnd = $derived(
		Math.min(totalRows, Math.ceil((scrollTop + CONTAINER_HEIGHT) / ROW_HEIGHT) + OVERSCAN)
	);
	const topPad = $derived(virtualStart * ROW_HEIGHT);
	const bottomPad = $derived(Math.max(0, (totalRows - virtualEnd) * ROW_HEIGHT));
	const visibleRows = $derived(filteredData.slice(virtualStart, virtualEnd));

	$effect(() => {
		if (!scrollContainer || !mirrorBar) return;

		const obs = new ResizeObserver(() => {
			tableScrollWidth = scrollContainer!.scrollWidth;
		});
		obs.observe(scrollContainer);

		const onScroll = () => {
			scrollTop = scrollContainer!.scrollTop;
			mirrorBar!.scrollLeft = scrollContainer!.scrollLeft;
		};
		const onMirrorScroll = () => {
			scrollContainer!.scrollLeft = mirrorBar!.scrollLeft;
		};

		scrollContainer.addEventListener('scroll', onScroll);
		mirrorBar.addEventListener('scroll', onMirrorScroll);

		return () => {
			obs.disconnect();
			scrollContainer!.removeEventListener('scroll', onScroll);
			mirrorBar!.removeEventListener('scroll', onMirrorScroll);
		};
	});

	const placeholderRows = Array.from({ length: 5 }, (_, i) => i);

	function formatColumnName(column: string) {
		if (columnLabels[column]) return columnLabels[column];
		return column
			.split('_')
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	}

	function stringifyValue(value: unknown) {
		if (Array.isArray(value)) return value.join(', ');
		if (typeof value === 'object' && value) return JSON.stringify(value);
		return value?.toString() ?? '';
	}

	function toggleColumn(col: string) {
		const next = new Set(hiddenColumns);
		if (next.has(col)) {
			next.delete(col);
		} else {
			next.add(col);
		}
		hiddenColumns = next;
	}

	function clearFilters() {
		columnFilters = {};
	}
</script>

<svelte:window
	onclick={(e) => {
		if (!(e.target as HTMLElement).closest('[data-columns-menu]')) columnsMenuOpen = false;
	}}
/>

<section class="rounded-lg bg-white p-6 shadow">
	<div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
		<div class="flex items-center gap-3">
			<button
				type="button"
				class="text-gray-500 hover:text-gray-700"
				onclick={() => (isCollapsed = !isCollapsed)}
				aria-label={isCollapsed ? 'Expand table' : 'Collapse table'}
			>
				<svg
					class="h-6 w-6 transition-transform duration-200"
					class:rotate-180={isCollapsed}
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
				</svg>
			</button>
			<div>
				<h2 class="text-xl font-semibold text-gray-900">Results</h2>
				<span class="text-sm text-gray-600">
					{#if fetching}
						{results?.data?.length
							? `${results.data.length} rows — fetching more…`
							: 'Querying ALMA TAP…'}
					{:else if hasActiveFilters}
						{filteredData.length} of {results?.data?.length ?? 0} rows (filtered)
					{:else}
						{results?.count ? `${results.count} rows` : 'No data yet'}
					{/if}
				</span>
				{#if fetching}
					<span
						class="h-3 w-3 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"
						aria-hidden="true"
					></span>
				{/if}
			</div>
		</div>
		<div class="flex flex-wrap gap-2">
			<!-- Column visibility toggle -->
			<div class="relative" data-columns-menu>
				<button
					type="button"
					class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
					onclick={() => (columnsMenuOpen = !columnsMenuOpen)}
				>
					Columns ({tableColumns.length}/{menuColumns.length})
				</button>
				{#if columnsMenuOpen}
					<div
						class="absolute right-0 z-30 mt-1 max-h-72 w-56 overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg"
					>
						<div class="flex items-center justify-between border-b border-gray-100 px-3 py-2">
							<span class="text-xs font-semibold uppercase tracking-wide text-gray-500"
								>Show / Hide</span
							>
							<div class="flex gap-2">
								<button
									type="button"
									class="text-xs text-blue-600 hover:underline"
									onclick={() => (hiddenColumns = new Set())}>All</button
								>
								<button
									type="button"
									class="text-xs text-blue-600 hover:underline"
									onclick={() => (hiddenColumns = new Set(menuColumns))}>None</button
								>
							</div>
						</div>
						{#each menuColumns as col}
							<label
								class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
								class:text-gray-400={!dataColumnSet.has(col)}
							>
								<input
									type="checkbox"
									checked={!hiddenColumns.has(col)}
									onchange={() => toggleColumn(col)}
									class="rounded border-gray-300"
								/>
								{formatColumnName(col)}
								{#if !dataColumnSet.has(col)}
									<span class="ml-auto text-xs text-gray-300">—</span>
								{/if}
							</label>
						{/each}
					</div>
				{/if}
			</div>
			{#if hasActiveFilters}
				<button
					type="button"
					class="rounded-md border border-amber-300 px-4 py-2 text-sm font-medium text-amber-700 hover:bg-amber-50"
					onclick={clearFilters}
				>
					Clear Filters
				</button>
			{/if}
			<button
				type="button"
				class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed"
				onclick={onClear}
				disabled={loading || !results}
			>
				Clear Metadata
			</button>
			<button
				type="button"
				class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
				onclick={onLoad}
			>
				Load Metadata
			</button>
			{#if results?.data?.length}
				<button
					type="button"
					class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300"
					disabled={saving}
					onclick={onSave}
				>
					{saving ? 'Saving...' : 'Save Metadata'}
				</button>
			{/if}
		</div>
	</div>

	{#if !isCollapsed}
		<!-- Top mirror scrollbar: scrolls horizontally in sync with the table below -->
		<div
			bind:this={mirrorBar}
			class="mt-4 overflow-x-scroll overflow-y-hidden border-b border-gray-100"
			style="height: 14px;"
		>
			<div style="height: 1px; width: {tableScrollWidth}px;"></div>
		</div>

		<!-- Main scroll container: vertical (capped height) + horizontal -->
		<div bind:this={scrollContainer} class="overflow-auto" style="max-height: {CONTAINER_HEIGHT}px;">
			<table class="divide-y divide-gray-200 text-sm" style="min-width: max-content; width: 100%;">
				<thead class="sticky top-0 z-10">
					<!-- Column header row -->
					<tr class="bg-gray-50">
						{#each tableColumns as column}
							<th
								scope="col"
								class="whitespace-nowrap px-4 py-2 text-left font-semibold text-gray-700"
							>
								{formatColumnName(column)}
							</th>
						{/each}
					</tr>
					<!-- Filter row — only shown when there is data -->
					{#if results?.data?.length}
						<tr class="bg-white shadow-sm">
							{#each tableColumns as column}
								<th scope="col" class="px-2 py-1">
									<input
										type="text"
										placeholder="filter…"
										bind:value={columnFilters[column]}
										class="w-full min-w-16 rounded border border-gray-200 px-1.5 py-0.5 text-xs font-normal text-gray-700 placeholder-gray-300 focus:border-blue-400 focus:outline-none"
										class:border-blue-400={columnFilters[column]?.length > 0}
										class:bg-blue-50={columnFilters[column]?.length > 0}
									/>
								</th>
							{/each}
						</tr>
					{/if}
				</thead>
				<tbody class="bg-white">
					{#if !loading && results?.data?.length}
						{#if topPad > 0}
							<tr style="height: {topPad}px;" aria-hidden="true">
								<td colspan={tableColumns.length}></td>
							</tr>
						{/if}
						{#each visibleRows as row}
							<tr class="border-b border-gray-100 hover:bg-gray-50" style="height: {ROW_HEIGHT}px;">
								{#each tableColumns as column}
									{@const displayValue = stringifyValue(row[column])}
									<td class="px-4 align-middle">
										<div
											class="whitespace-nowrap text-gray-800"
											style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;"
											title={displayValue}
										>
											{displayValue}
										</div>
									</td>
								{/each}
							</tr>
						{/each}
						{#if bottomPad > 0}
							<tr style="height: {bottomPad}px;" aria-hidden="true">
								<td colspan={tableColumns.length}></td>
							</tr>
						{/if}
					{:else if !loading && hasActiveFilters}
						<tr>
							<td colspan={tableColumns.length} class="px-4 py-8 text-center text-sm text-gray-400">
								No rows match the current filters.
							</td>
						</tr>
					{:else}
						{#each placeholderRows as _}
							<tr class="animate-pulse border-b border-gray-100">
								{#each tableColumns as _}
									<td class="px-4 py-3">
										<div class="h-4 w-32 rounded bg-gray-200/80" aria-hidden="true"></div>
									</td>
								{/each}
							</tr>
						{/each}
					{/if}
				</tbody>
			</table>
		</div>
	{/if}
</section>
