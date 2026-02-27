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

	// Mirror-scroll: top scrollbar synced to the real scroll container
	let scrollContainer = $state<HTMLDivElement | null>(null);
	let mirrorBar = $state<HTMLDivElement | null>(null);
	let tableScrollWidth = $state(0);

	$effect(() => {
		if (!scrollContainer || !mirrorBar) return;

		const obs = new ResizeObserver(() => {
			tableScrollWidth = scrollContainer!.scrollWidth;
		});
		obs.observe(scrollContainer);

		const onScroll = () => { mirrorBar!.scrollLeft = scrollContainer!.scrollLeft; };
		const onMirrorScroll = () => { scrollContainer!.scrollLeft = mirrorBar!.scrollLeft; };

		scrollContainer.addEventListener('scroll', onScroll);
		mirrorBar.addEventListener('scroll', onMirrorScroll);

		return () => {
			obs.disconnect();
			scrollContainer!.removeEventListener('scroll', onScroll);
			mirrorBar!.removeEventListener('scroll', onMirrorScroll);
		};
	});

	const columnLabels: Record<string, string> = {
		ALMA_source_name: 'Source Name',
		Band: 'Band',
		antenna_arrays: 'Array',
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
		group_ous_uid: 'Group OUS UID'
	};

	// The 10 key fields shown first; remaining columns follow in data order
	const preferredOrder = [
		'ALMA_source_name',
		'Band',
		'antenna_arrays',
		'Ang.res.',
		'Obs.date',
		'Project_abstract',
		'science_keyword',
		'scientific_category',
		'QA2_status',
		'Type',
	];

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

	const tableColumns = $derived.by(() => {
		const data = results?.data;
		if (!data || data.length === 0) return preferredOrder;
		const allKeys = Object.keys(data[0]);
		const leading = preferredOrder.filter((c) => allKeys.includes(c));
		const rest = allKeys.filter((c) => !preferredOrder.includes(c));
		return [...leading, ...rest];
	});
</script>

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
						{results?.data?.length ? `${results.data.length} rows — fetching more…` : 'Querying ALMA TAP…'}
					{:else}
						{results?.count ? `${results.count} rows` : 'No data yet'}
					{/if}
				</span>
				{#if fetching}
					<span class="h-3 w-3 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" aria-hidden="true"></span>
				{/if}
			</div>
		</div>
		<div class="flex flex-wrap gap-2">
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
		<div
			bind:this={scrollContainer}
			class="overflow-auto"
			style="max-height: 480px;"
		>
			<table class="divide-y divide-gray-200 text-sm" style="min-width: max-content; width: 100%;">
				<thead class="sticky top-0 z-10 bg-gray-50">
					<tr>
						{#each tableColumns as column}
							<th
								scope="col"
								class="whitespace-nowrap px-4 py-2 text-left font-semibold text-gray-700"
							>
								{formatColumnName(column)}
							</th>
						{/each}
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100 bg-white">
					{#if !loading && results?.data?.length}
						{#each results.data as row}
							<tr class="hover:bg-gray-50">
								{#each tableColumns as column}
									{@const displayValue = stringifyValue(row[column])}
									<td class="px-4 py-2 align-top">
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
					{:else}
						{#each placeholderRows as _}
							<tr class="animate-pulse">
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
