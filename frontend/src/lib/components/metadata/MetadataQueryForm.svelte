<script lang="ts">
	import type { MetadataQuery } from '$lib/api/metadata';

	interface Props {
		scienceTypes: { keywords: string[]; categories: string[] } | null;
		loading: boolean;
		onSubmit: (query: MetadataQuery) => void;
	}

	let { scienceTypes, loading, onSubmit }: Props = $props();

	const bandOptions = [3, 4, 5, 6, 7, 8, 9, 10];
	const qa2Options = ['T', 'F'];

	// Multi-select state for keywords and categories
	let selectedKeywords = $state<string[]>([]);
	let selectedCategories = $state<string[]>([]);
	let keywordSearch = $state('');
	let categorySearch = $state('');
	let keywordDropdownOpen = $state(false);
	let categoryDropdownOpen = $state(false);

	const filteredKeywords = $derived(
		(scienceTypes?.keywords ?? []).filter(
			(k) => k.toLowerCase().includes(keywordSearch.toLowerCase())
		)
	);

	const filteredCategories = $derived(
		(scienceTypes?.categories ?? []).filter(
			(c) => c.toLowerCase().includes(categorySearch.toLowerCase())
		)
	);

	function toggleKeyword(keyword: string) {
		if (selectedKeywords.includes(keyword)) {
			selectedKeywords = selectedKeywords.filter((k) => k !== keyword);
		} else {
			selectedKeywords = [...selectedKeywords, keyword];
		}
	}

	function toggleCategory(category: string) {
		if (selectedCategories.includes(category)) {
			selectedCategories = selectedCategories.filter((c) => c !== category);
		} else {
			selectedCategories = [...selectedCategories, category];
		}
	}

	function clearKeywords() {
		selectedKeywords = [];
	}

	function clearCategories() {
		selectedCategories = [];
	}

	function handleSubmit(event: Event) {
		event.preventDefault();
		const formData = new FormData(event.currentTarget as HTMLFormElement);
		const query: MetadataQuery = {};

		const sourceName = formData.get('source_name') as string;
		if (sourceName?.trim()) query.source_name = sourceName.trim();

		if (selectedKeywords.length) query.science_keyword = selectedKeywords;
		if (selectedCategories.length) query.scientific_category = selectedCategories;

		const bands = formData
			.getAll('bands')
			.map((value) => Number(value))
			.filter((value) => Number.isFinite(value));
		if (bands.length) query.bands = bands;

		const antennaArrays = formData.get('antenna_arrays') as string;
		if (antennaArrays?.trim()) query.antenna_arrays = antennaArrays.trim();

		const angResMin = formData.get('ang_res_min');
		const angResMax = formData.get('ang_res_max');
		if (angResMin && angResMax) {
			const min = Number(angResMin);
			const max = Number(angResMax);
			if (Number.isFinite(min) && Number.isFinite(max)) {
				query.angular_resolution_range = [min, max];
			}
		}

		const obsDateMin = formData.get('obs_date_min') as string;
		const obsDateMax = formData.get('obs_date_max') as string;
		if (obsDateMin && obsDateMax) {
			query.observation_date_range = [obsDateMin, obsDateMax];
		}

		const qa2Status = formData.getAll('qa2_status').filter(Boolean) as string[];
		if (qa2Status.length) query.qa2_status = qa2Status;

		const obsType = formData.get('obs_type') as string;
		if (obsType?.trim()) query.obs_type = obsType.trim();

		const fovMin = formData.get('fov_min');
		const fovMax = formData.get('fov_max');
		if (fovMin && fovMax) {
			const min = Number(fovMin);
			const max = Number(fovMax);
			if (Number.isFinite(min) && Number.isFinite(max)) {
				query.fov_range = [min, max];
			}
		}

		const timeMin = formData.get('time_min');
		const timeMax = formData.get('time_max');
		if (timeMin && timeMax) {
			const min = Number(timeMin);
			const max = Number(timeMax);
			if (Number.isFinite(min) && Number.isFinite(max)) {
				query.time_resolution_range = [min, max];
			}
		}

		const freqMin = formData.get('freq_min');
		const freqMax = formData.get('freq_max');
		if (freqMin && freqMax) {
			const min = Number(freqMin);
			const max = Number(freqMax);
			if (Number.isFinite(min) && Number.isFinite(max)) {
				query.frequency_range = [min, max];
			}
		}

		onSubmit(query);
	}
</script>

<svelte:window
	onclick={(e) => {
		if (!(e.target as HTMLElement).closest('[data-dropdown="keyword"]'))
			keywordDropdownOpen = false;
		if (!(e.target as HTMLElement).closest('[data-dropdown="category"]'))
			categoryDropdownOpen = false;
	}}
/>

<form onsubmit={handleSubmit} class="space-y-4 rounded-lg bg-white p-6 shadow">
	<div class="flex items-center justify-between">
		<h2 class="text-xl font-semibold text-gray-900">Query Builder</h2>
		<button
			type="submit"
			disabled={loading}
			class="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:bg-gray-400"
		>
			{loading ? 'Running...' : 'Run Query'}
		</button>
	</div>

	<!-- Source Name -->
	<div>
		<label class="block">
			<span class="text-sm font-medium text-gray-700">Source Name</span>
			<input
				type="text"
				name="source_name"
				placeholder="e.g. NGC 1068"
				class="mt-1 w-full rounded-md border border-gray-300 p-2"
			/>
		</label>
	</div>

	<!-- Science Keywords & Categories -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">

		<!-- Science Keywords multi-select -->
		<div data-dropdown="keyword">
			<span class="text-sm font-medium text-gray-700">Science Keywords</span>

			<!-- Selected chips -->
			{#if selectedKeywords.length > 0}
				<div class="mt-1 flex flex-wrap gap-1">
					{#each selectedKeywords as kw}
						<span class="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-800">
							{kw}
							<button
								type="button"
								onclick={() => toggleKeyword(kw)}
								class="ml-0.5 text-blue-500 hover:text-blue-700"
								aria-label="Remove {kw}"
							>✕</button>
						</span>
					{/each}
					<button
						type="button"
						onclick={clearKeywords}
						class="text-xs text-gray-400 hover:text-gray-600 underline"
					>Clear all</button>
				</div>
			{/if}

			<!-- Search input -->
			<div class="relative mt-1">
				<input
					type="text"
					placeholder={scienceTypes ? 'Search keywords…' : 'Loading…'}
					disabled={!scienceTypes}
					bind:value={keywordSearch}
					onfocus={() => (keywordDropdownOpen = true)}
					class="w-full rounded-md border border-gray-300 p-2 text-sm disabled:bg-gray-50"
				/>

				{#if keywordDropdownOpen && scienceTypes}
					<div class="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
						{#if filteredKeywords.length === 0}
							<p class="px-3 py-2 text-sm text-gray-400">No results</p>
						{:else}
							{#each filteredKeywords as kw}
								<label class="flex cursor-pointer items-center gap-2 px-3 py-1.5 hover:bg-gray-50 text-sm">
									<input
										type="checkbox"
										checked={selectedKeywords.includes(kw)}
										onchange={() => toggleKeyword(kw)}
										class="rounded border-gray-300"
									/>
									{kw}
								</label>
							{/each}
						{/if}
					</div>
				{/if}
			</div>
			<p class="mt-0.5 text-xs text-gray-400">{selectedKeywords.length} selected</p>
		</div>

		<!-- Science Categories multi-select -->
		<div data-dropdown="category">
			<span class="text-sm font-medium text-gray-700">Science Category</span>

			<!-- Selected chips -->
			{#if selectedCategories.length > 0}
				<div class="mt-1 flex flex-wrap gap-1">
					{#each selectedCategories as cat}
						<span class="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-xs text-purple-800">
							{cat}
							<button
								type="button"
								onclick={() => toggleCategory(cat)}
								class="ml-0.5 text-purple-500 hover:text-purple-700"
								aria-label="Remove {cat}"
							>✕</button>
						</span>
					{/each}
					<button
						type="button"
						onclick={clearCategories}
						class="text-xs text-gray-400 hover:text-gray-600 underline"
					>Clear all</button>
				</div>
			{/if}

			<!-- Search input -->
			<div class="relative mt-1">
				<input
					type="text"
					placeholder={scienceTypes ? 'Search categories…' : 'Loading…'}
					disabled={!scienceTypes}
					bind:value={categorySearch}
					onfocus={() => (categoryDropdownOpen = true)}
					class="w-full rounded-md border border-gray-300 p-2 text-sm disabled:bg-gray-50"
				/>

				{#if categoryDropdownOpen && scienceTypes}
					<div class="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
						{#if filteredCategories.length === 0}
							<p class="px-3 py-2 text-sm text-gray-400">No results</p>
						{:else}
							{#each filteredCategories as cat}
								<label class="flex cursor-pointer items-center gap-2 px-3 py-1.5 hover:bg-gray-50 text-sm">
									<input
										type="checkbox"
										checked={selectedCategories.includes(cat)}
										onchange={() => toggleCategory(cat)}
										class="rounded border-gray-300"
									/>
									{cat}
								</label>
							{/each}
						{/if}
					</div>
				{/if}
			</div>
			<p class="mt-0.5 text-xs text-gray-400">{selectedCategories.length} selected</p>
		</div>
	</div>

	<!-- Bands -->
	<div>
		<span class="text-sm font-medium text-gray-700">Band</span>
		<div class="mt-2 flex flex-wrap gap-2">
			{#each bandOptions as band}
				<label class="inline-flex items-center space-x-1 text-sm text-gray-700">
					<input type="checkbox" name="bands" value={band} class="rounded border-gray-300" />
					<span>Band {band}</span>
				</label>
			{/each}
		</div>
	</div>

	<!-- Array (Antenna Arrays) -->
	<div>
		<label class="block">
			<span class="text-sm font-medium text-gray-700">Array Configuration</span>
			<input
				type="text"
				name="antenna_arrays"
				placeholder="e.g. C43-3"
				class="mt-1 w-full rounded-md border border-gray-300 p-2"
			/>
		</label>
	</div>

	<!-- Angular Resolution & Observation Date -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="block text-sm font-medium text-gray-700">Angular Resolution (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.01"
					name="ang_res_min"
					placeholder="Min"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
				<input
					type="number"
					step="0.01"
					name="ang_res_max"
					placeholder="Max"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
			</div>
		</div>

		<div>
			<span class="block text-sm font-medium text-gray-700">Observation Date</span>
			<div class="mt-1 flex gap-2">
				<input
					type="date"
					name="obs_date_min"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
				<input
					type="date"
					name="obs_date_max"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
			</div>
		</div>
	</div>

	<!-- QA2 Status & Type -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="text-sm font-medium text-gray-700">QA2 Status</span>
			<div class="mt-2 flex gap-4">
				{#each qa2Options as qa2}
					<label class="inline-flex items-center space-x-1 text-sm text-gray-700">
						<input type="checkbox" name="qa2_status" value={qa2} class="rounded border-gray-300" />
						<span>{qa2 === 'T' ? 'Pass (T)' : 'Fail (F)'}</span>
					</label>
				{/each}
			</div>
		</div>

		<div>
			<label class="block">
				<span class="text-sm font-medium text-gray-700">Type</span>
				<input
					type="text"
					name="obs_type"
					placeholder="e.g. S"
					class="mt-1 w-full rounded-md border border-gray-300 p-2"
				/>
			</label>
		</div>
	</div>

	<!-- FOV, Time Resolution, Frequency -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<span class="block text-sm font-medium text-gray-700">FOV (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="fov_min"
					placeholder="Min"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="fov_max"
					placeholder="Max"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
			</div>
		</div>

		<div>
			<span class="block text-sm font-medium text-gray-700">Time Resolution (s)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="time_min"
					placeholder="Min"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="time_max"
					placeholder="Max"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
			</div>
		</div>

		<div>
			<span class="block text-sm font-medium text-gray-700">Frequency (GHz)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="freq_min"
					placeholder="Min"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="freq_max"
					placeholder="Max"
					class="w-full rounded-md border border-gray-300 p-2"
				/>
			</div>
		</div>
	</div>
</form>
