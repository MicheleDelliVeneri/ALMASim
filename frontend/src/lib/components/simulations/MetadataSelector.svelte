<script lang="ts">
	import type { MetadataResponse } from '#lib/api/metadata.js';
	import { deriveArrayType } from '#lib/utils/observationPlan.js';

	interface Props {
		metadata: MetadataResponse | null;
		selectedIndices: number[];
		onSelect: (indices: number[]) => void;
		getRowValue: (row: Record<string, unknown>, key: string) => string;
		getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
	}

	let { metadata, selectedIndices, onSelect, getRowValue, getRowNumber }: Props = $props();

	let isCollapsed = $state(true);

	function toggleSelection(index: number) {
		const currentIndices = [...selectedIndices];
		const existingIndex = currentIndices.indexOf(index);

		if (existingIndex > -1) {
			// Remove if already selected
			currentIndices.splice(existingIndex, 1);
		} else {
			// Add if not selected
			currentIndices.push(index);
		}

		onSelect(currentIndices);
	}

	function selectAll() {
		if (!metadata?.data) return;
		const allIndices = metadata.data.map((_, index) => index);
		onSelect(allIndices);
	}

	function clearAll() {
		onSelect([]);
	}
</script>

<section class="rounded-lg bg-surface p-6 shadow-panel">
	<div class="mb-4 flex items-center justify-between">
		<div class="flex items-center gap-3">
			<button
				type="button"
				class="text-fg-dim hover:text-gray-700"
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
				<h2 class="text-xl font-semibold text-fg">Select Metadata Rows</h2>
			<p class="mt-1 text-sm text-fg-muted">
				{metadata?.count
					? `${selectedIndices.length} of ${metadata?.count} rows selected`
					: 'No metadata loaded. Go to Metadata page to query or load data.'}
			</p>
			</div>
		</div>
		<div class="flex items-center gap-2">
			{#if metadata?.data && metadata.data.length > 0}
				<button
					type="button"
					onclick={selectAll}
					class="text-sm font-medium text-accent hover:text-accent"
				>
					Select All
				</button>
				<span class="text-gray-300">|</span>
				<button
					type="button"
					onclick={clearAll}
					class="text-sm font-medium text-accent hover:text-accent"
				>
					Clear
				</button>
				<span class="text-gray-300">|</span>
			{/if}
			<a href="/metadata" class="text-sm font-medium text-accent hover:text-accent">
				Go to Metadata →
			</a>
		</div>
	</div>

	{#if !isCollapsed}
		{#if metadata?.data && metadata.data.length > 0}
		<div class="max-h-[28rem] overflow-auto">
			<table class="min-w-full divide-y divide-line text-sm">
				<thead class="bg-surface">
					<tr>
						<th class="w-12 px-4 py-2 text-left font-semibold text-fg-muted">Select</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">Source Name</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">RA (deg)</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">Dec (deg)</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">Band</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">Array Type</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">FOV (arcsec)</th>
						<th class="px-4 py-2 text-left font-semibold text-fg-muted">Freq (GHz)</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-line bg-surface">
					{#each metadata.data as row, index}
						{@const isSelected = selectedIndices.includes(index)}
						{@const raVal = getRowNumber(row, 'RA') ?? getRowNumber(row, 'ra')}
						{@const decVal = getRowNumber(row, 'Dec') ?? getRowNumber(row, 'dec')}
						{@const fovVal = getRowNumber(row, 'FOV') ?? getRowNumber(row, 'fov')}
						{@const freqVal = getRowNumber(row, 'Freq') ?? getRowNumber(row, 'freq')}
						<tr
							class={`cursor-pointer transition-colors ${
								isSelected ? 'bg-accent-soft hover:bg-accent-soft' : 'hover:bg-hover'
							}`}
							onclick={() => toggleSelection(index)}
						>
							<td class="px-4 py-2">
								<input
									type="checkbox"
									checked={isSelected}
									onchange={() => toggleSelection(index)}
									class="rounded text-accent"
									onclick={(e) => e.stopPropagation()}
								/>
							</td>
							<td class="px-4 py-2 text-fg">
								{getRowValue(row, 'ALMA_source_name') || getRowValue(row, 'source_name')}
							</td>
							<td class="px-4 py-2 text-fg">
								{raVal !== null ? raVal.toFixed(4) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-fg">
								{decVal !== null ? decVal.toFixed(4) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-fg">
								{getRowValue(row, 'Band') || getRowValue(row, 'band')}
							</td>
							<td class="px-4 py-2 text-fg">
								{getRowValue(row, 'Array_type') ||
									deriveArrayType(
										getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array')
									) ||
									'N/A'}
							</td>
							<td class="px-4 py-2 text-fg">
								{fovVal !== null ? fovVal.toFixed(2) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-fg">
								{freqVal !== null ? freqVal.toFixed(2) : 'N/A'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		{:else}
			<div class="py-8 text-center text-fg-dim">
				<p>No metadata available. Please query or load metadata first.</p>
				<a href="/metadata" class="mt-2 inline-block text-accent hover:text-accent">
					Go to Metadata page
				</a>
			</div>
		{/if}
	{/if}
</section>
