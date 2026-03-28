<script lang="ts">
	import type { MetadataResponse } from '$lib/api/metadata';
	import { deriveArrayType } from '$lib/utils/observationPlan';

	interface Props {
		metadata: MetadataResponse | null;
		selectedIndices: number[];
		onSelect: (indices: number[]) => void;
		getRowValue: (row: Record<string, unknown>, key: string) => string;
		getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
	}

	let { metadata, selectedIndices, onSelect, getRowValue, getRowNumber }: Props = $props();

	let isCollapsed = $state(false);

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

<section class="rounded-lg bg-white p-6 shadow-md">
	<div class="mb-4 flex items-center justify-between">
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
				<h2 class="text-xl font-semibold text-gray-900">Select Metadata Rows</h2>
			<p class="mt-1 text-sm text-gray-600">
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
					class="text-sm font-medium text-blue-600 hover:text-blue-800"
				>
					Select All
				</button>
				<span class="text-gray-300">|</span>
				<button
					type="button"
					onclick={clearAll}
					class="text-sm font-medium text-blue-600 hover:text-blue-800"
				>
					Clear
				</button>
				<span class="text-gray-300">|</span>
			{/if}
			<a href="/metadata" class="text-sm font-medium text-blue-600 hover:text-blue-800">
				Go to Metadata →
			</a>
		</div>
	</div>

	{#if !isCollapsed}
		{#if metadata?.data && metadata.data.length > 0}
		<div class="overflow-x-auto">
			<table class="min-w-full divide-y divide-gray-200 text-sm">
				<thead class="bg-gray-50">
					<tr>
						<th class="w-12 px-4 py-2 text-left font-semibold text-gray-700">Select</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">Source Name</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">RA (deg)</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">Dec (deg)</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">Band</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">Array Type</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">FOV (arcsec)</th>
						<th class="px-4 py-2 text-left font-semibold text-gray-700">Freq (GHz)</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100 bg-white">
					{#each metadata.data as row, index}
						{@const isSelected = selectedIndices.includes(index)}
						{@const raVal = getRowNumber(row, 'RA') ?? getRowNumber(row, 'ra')}
						{@const decVal = getRowNumber(row, 'Dec') ?? getRowNumber(row, 'dec')}
						{@const fovVal = getRowNumber(row, 'FOV') ?? getRowNumber(row, 'fov')}
						{@const freqVal = getRowNumber(row, 'Freq') ?? getRowNumber(row, 'freq')}
						<tr
							class={`cursor-pointer transition-colors ${
								isSelected ? 'bg-blue-50 hover:bg-blue-100' : 'hover:bg-gray-50'
							}`}
							onclick={() => toggleSelection(index)}
						>
							<td class="px-4 py-2">
								<input
									type="checkbox"
									checked={isSelected}
									onchange={() => toggleSelection(index)}
									class="rounded text-blue-600"
									onclick={(e) => e.stopPropagation()}
								/>
							</td>
							<td class="px-4 py-2 text-gray-800">
								{getRowValue(row, 'ALMA_source_name') || getRowValue(row, 'source_name')}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{raVal !== null ? raVal.toFixed(4) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{decVal !== null ? decVal.toFixed(4) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{getRowValue(row, 'Band') || getRowValue(row, 'band')}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{getRowValue(row, 'Array_type') ||
									deriveArrayType(
										getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array')
									) ||
									'N/A'}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{fovVal !== null ? fovVal.toFixed(2) : 'N/A'}
							</td>
							<td class="px-4 py-2 text-gray-800">
								{freqVal !== null ? freqVal.toFixed(2) : 'N/A'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		{:else}
			<div class="py-8 text-center text-gray-500">
				<p>No metadata available. Please query or load metadata first.</p>
				<a href="/metadata" class="mt-2 inline-block text-blue-600 hover:text-blue-800">
					Go to Metadata page
				</a>
			</div>
		{/if}
	{/if}
</section>
