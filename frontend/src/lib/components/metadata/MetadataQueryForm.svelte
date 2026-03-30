<script lang="ts">
	import type { MetadataQuery } from '$lib/api/metadata';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('components/metadata/MetadataQueryForm');

	interface Props {
		scienceTypes: { keywords: string[]; categories: string[] } | null;
		loading: boolean;
		onSubmit: (query: MetadataQuery) => void;
	}

	let { scienceTypes, loading, onSubmit }: Props = $props();

	const bandOptions = [3, 4, 5, 6, 7, 8, 9, 10];
	const arrayTypeOptions = ['12m', '7m', 'TP'];
	// Cycle N → year 2012+N. Cycles 1-12 cover 2013-2024 (Cycle 0 = 2012, pilot).
	const cycleOptions = Array.from({ length: 13 }, (_, i) => ({
		cycle: i,
		year: 2012 + i,
		label: `Cycle ${i} (${2012 + i})`,
	}));
	let selectedCycles = $state<number[]>([]);
	const qa2Options = [
		{ value: 'Pass', label: 'Pass' },
		{ value: 'SemiPass', label: 'Semi-Pass' },
		{ value: 'Fail', label: 'Fail' },
	];

	// Inclusion multi-select state
	let selectedKeywords = $state<string[]>([]);
	let selectedCategories = $state<string[]>([]);
	let keywordSearch = $state('');
	let categorySearch = $state('');
	let keywordDropdownOpen = $state(false);
	let categoryDropdownOpen = $state(false);

	// Exclusion state
	let excludeSection = $state(false);
	let excludeSelectedKeywords = $state<string[]>([]);
	let excludeSelectedCategories = $state<string[]>([]);
	let excludeKeywordSearch = $state('');
	let excludeCategorySearch = $state('');
	let excludeKeywordDropdownOpen = $state(false);
	let excludeCategoryDropdownOpen = $state(false);
	let excludeSolar = $state(false);
	let publicOnly = $state(true);
	let scienceOnly = $state(true);
	let excludeMosaic = $state(true);

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

	const filteredExcludeKeywords = $derived(
		(scienceTypes?.keywords ?? []).filter(
			(k) => k.toLowerCase().includes(excludeKeywordSearch.toLowerCase())
		)
	);

	const filteredExcludeCategories = $derived(
		(scienceTypes?.categories ?? []).filter(
			(c) => c.toLowerCase().includes(excludeCategorySearch.toLowerCase())
		)
	);

	function toggleKeyword(keyword: string) {
		selectedKeywords = selectedKeywords.includes(keyword)
			? selectedKeywords.filter((k) => k !== keyword)
			: [...selectedKeywords, keyword];
		logger.debug({ keyword, selected: selectedKeywords.includes(keyword) }, 'Science keyword toggled');
	}

	function toggleCategory(category: string) {
		selectedCategories = selectedCategories.includes(category)
			? selectedCategories.filter((c) => c !== category)
			: [...selectedCategories, category];
		logger.debug({ category, selected: selectedCategories.includes(category) }, 'Science category toggled');
	}

	function toggleExcludeKeyword(keyword: string) {
		excludeSelectedKeywords = excludeSelectedKeywords.includes(keyword)
			? excludeSelectedKeywords.filter((k) => k !== keyword)
			: [...excludeSelectedKeywords, keyword];
	}

	function toggleExcludeCategory(category: string) {
		excludeSelectedCategories = excludeSelectedCategories.includes(category)
			? excludeSelectedCategories.filter((c) => c !== category)
			: [...excludeSelectedCategories, category];
	}

	function handleSubmit(event: Event) {
		event.preventDefault();
		const formData = new FormData(event.currentTarget as HTMLFormElement);
		const query: MetadataQuery = {};

		const sourceName = formData.get('source_name') as string;
		if (sourceName?.trim()) query.source_name = sourceName.trim();

		if (selectedKeywords.length) query.science_keyword = selectedKeywords;
		if (selectedCategories.length) query.scientific_category = selectedCategories;

		const bands = formData.getAll('bands').map(Number).filter(Number.isFinite);
		if (bands.length) query.bands = bands;

		const arrayTypes = formData.getAll('array_type') as string[];
		if (arrayTypes.length) query.array_type = arrayTypes;

		const arrayConfig = formData.get('array_configuration') as string;
		if (arrayConfig?.trim()) {
			query.array_configuration = arrayConfig.split(',').map((s) => s.trim()).filter(Boolean);
		}

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

		const obsTypes = formData.getAll('obs_type') as string[];
		if (obsTypes.length) query.obs_type = obsTypes;

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

		// Exclusion filters
		if (excludeSelectedKeywords.length) query.exclude_science_keyword = excludeSelectedKeywords;
		if (excludeSelectedCategories.length) query.exclude_scientific_category = excludeSelectedCategories;

		const excludeSourceName = formData.get('exclude_source_name') as string;
		if (excludeSourceName?.trim()) {
			query.exclude_source_name = excludeSourceName.split(',').map((s) => s.trim()).filter(Boolean);
		}

		const excludeObsType = formData.get('exclude_obs_type') as string;
		if (excludeObsType?.trim()) {
			query.exclude_obs_type = excludeObsType.split(',').map((s) => s.trim()).filter(Boolean);
		}

		if (excludeSolar) query.exclude_solar = true;

		if (selectedCycles.length) query.cycles = selectedCycles;

		query.public_only = publicOnly;
		query.science_only = scienceOnly;
		query.exclude_mosaic = excludeMosaic;

		logger.info(
			{
				sourceName: query.source_name,
				bands: query.bands,
				cycles: query.cycles,
				keywordCount: (query.science_keyword ?? []).length,
				categoryCount: (query.scientific_category ?? []).length,
				publicOnly,
				scienceOnly,
			},
			'Query form submitted'
		);
		onSubmit(query);
	}
</script>

<svelte:window
	onclick={(e) => {
		if (!(e.target as HTMLElement).closest('[data-dropdown="keyword"]'))
			keywordDropdownOpen = false;
		if (!(e.target as HTMLElement).closest('[data-dropdown="category"]'))
			categoryDropdownOpen = false;
		if (!(e.target as HTMLElement).closest('[data-dropdown="ex-keyword"]'))
			excludeKeywordDropdownOpen = false;
		if (!(e.target as HTMLElement).closest('[data-dropdown="ex-category"]'))
			excludeCategoryDropdownOpen = false;
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

	<!-- Quick-filter toggles -->
	<div class="flex flex-wrap gap-x-6 gap-y-2">
		<label class="inline-flex items-center gap-2 text-sm text-gray-700">
			<input type="checkbox" bind:checked={publicOnly} class="rounded border-gray-300" />
			<span>Public data only</span>
		</label>
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
			{#if selectedKeywords.length > 0}
				<div class="mt-1 flex flex-wrap gap-1">
					{#each selectedKeywords as kw}
						<span class="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-800">
							{kw}
							<button type="button" onclick={() => toggleKeyword(kw)} class="ml-0.5 text-blue-500 hover:text-blue-700" aria-label="Remove {kw}">✕</button>
						</span>
					{/each}
					<button type="button" onclick={() => (selectedKeywords = [])} class="text-xs text-gray-400 hover:text-gray-600 underline">Clear all</button>
				</div>
			{/if}
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
									<input type="checkbox" checked={selectedKeywords.includes(kw)} onchange={() => toggleKeyword(kw)} class="rounded border-gray-300" />
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
			{#if selectedCategories.length > 0}
				<div class="mt-1 flex flex-wrap gap-1">
					{#each selectedCategories as cat}
						<span class="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-xs text-purple-800">
							{cat}
							<button type="button" onclick={() => toggleCategory(cat)} class="ml-0.5 text-purple-500 hover:text-purple-700" aria-label="Remove {cat}">✕</button>
						</span>
					{/each}
					<button type="button" onclick={() => (selectedCategories = [])} class="text-xs text-gray-400 hover:text-gray-600 underline">Clear all</button>
				</div>
			{/if}
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
									<input type="checkbox" checked={selectedCategories.includes(cat)} onchange={() => toggleCategory(cat)} class="rounded border-gray-300" />
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

	<!-- ALMA Cycle -->
	<div>
		<div class="flex items-center justify-between">
			<span class="text-sm font-medium text-gray-700">ALMA Cycle</span>
			{#if selectedCycles.length > 0}
				<button
					type="button"
					onclick={() => (selectedCycles = [])}
					class="text-xs text-gray-400 hover:text-gray-600 underline"
				>Clear all</button>
			{/if}
		</div>
		<div class="mt-2 flex flex-wrap gap-2">
			{#each cycleOptions as opt}
				<label class="inline-flex items-center space-x-1 text-sm text-gray-700">
					<input
						type="checkbox"
						checked={selectedCycles.includes(opt.cycle)}
						onchange={() => {
							selectedCycles = selectedCycles.includes(opt.cycle)
								? selectedCycles.filter((c) => c !== opt.cycle)
								: [...selectedCycles, opt.cycle];
						}}
						class="rounded border-gray-300"
					/>
					<span>{opt.label}</span>
				</label>
			{/each}
		</div>
		<p class="mt-0.5 text-xs text-gray-400">
			{selectedCycles.length > 0
				? `Filtering by ${selectedCycles.length} cycle(s): proposal_id prefix ${selectedCycles.map((c) => `${2012 + c}.`).join(', ')}`
				: 'All cycles (no filter)'}
		</p>
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

	<!-- Array Type & Array Configuration -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="text-sm font-medium text-gray-700">Array Type</span>
			<div class="mt-2 flex gap-4">
				{#each arrayTypeOptions as atype}
					<label class="inline-flex items-center space-x-1 text-sm text-gray-700">
						<input type="checkbox" name="array_type" value={atype} class="rounded border-gray-300" />
						<span>{atype}</span>
					</label>
				{/each}
			</div>
		</div>
		<div>
			<label class="block">
				<span class="text-sm font-medium text-gray-700">Array Configuration</span>
				<input
					type="text"
					name="array_configuration"
					placeholder="e.g. C-1, C-2 (comma-separated)"
					class="mt-1 w-full rounded-md border border-gray-300 p-2"
				/>
			</label>
		</div>
	</div>

	<!-- Angular Resolution & Observation Date -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="block text-sm font-medium text-gray-700">Angular Resolution (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input type="number" step="0.01" name="ang_res_min" placeholder="Min" class="w-full rounded-md border border-gray-300 p-2" />
				<input type="number" step="0.01" name="ang_res_max" placeholder="Max" class="w-full rounded-md border border-gray-300 p-2" />
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-gray-700">Observation Date</span>
			<div class="mt-1 flex gap-2">
				<input type="date" name="obs_date_min" class="w-full rounded-md border border-gray-300 p-2" />
				<input type="date" name="obs_date_max" class="w-full rounded-md border border-gray-300 p-2" />
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
						<input type="checkbox" name="qa2_status" value={qa2.value} class="rounded border-gray-300" />
						<span>{qa2.label}</span>
					</label>
				{/each}
			</div>
		</div>
		<div>
			<div>
				<span class="text-sm font-medium text-gray-700">Project Type</span>
				<div class="mt-2 flex flex-wrap gap-3">
					{#each [['S', 'Science'], ['L', 'Large'], ['SV', 'Science Verification'], ['V', 'VLBI'], ['T', 'Target of Opportunity'], ['P', 'Phased Array']] as [val, lbl]}
						<label class="inline-flex items-center space-x-1 text-sm text-gray-700">
							<input type="checkbox" name="obs_type" value={val} class="rounded border-gray-300" />
							<span>{val} ({lbl})</span>
						</label>
					{/each}
				</div>
			</div>
			<div class="mt-2 flex gap-4">
				<label class="inline-flex items-center gap-1.5 text-sm text-gray-700">
					<input type="checkbox" bind:checked={scienceOnly} class="rounded border-gray-300" />
					<span>Science only</span>
				</label>
				<label class="inline-flex items-center gap-1.5 text-sm text-gray-700">
					<input type="checkbox" bind:checked={excludeMosaic} class="rounded border-gray-300" />
					<span>Exclude mosaics</span>
				</label>
			</div>
		</div>
	</div>

	<!-- FOV, Time Resolution, Frequency -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<span class="block text-sm font-medium text-gray-700">FOV (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input type="number" step="0.1" name="fov_min" placeholder="Min" class="w-full rounded-md border border-gray-300 p-2" />
				<input type="number" step="0.1" name="fov_max" placeholder="Max" class="w-full rounded-md border border-gray-300 p-2" />
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-gray-700">Time Resolution (s)</span>
			<div class="mt-1 flex gap-2">
				<input type="number" step="0.1" name="time_min" placeholder="Min" class="w-full rounded-md border border-gray-300 p-2" />
				<input type="number" step="0.1" name="time_max" placeholder="Max" class="w-full rounded-md border border-gray-300 p-2" />
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-gray-700">Frequency (GHz)</span>
			<div class="mt-1 flex gap-2">
				<input type="number" step="0.1" name="freq_min" placeholder="Min" class="w-full rounded-md border border-gray-300 p-2" />
				<input type="number" step="0.1" name="freq_max" placeholder="Max" class="w-full rounded-md border border-gray-300 p-2" />
			</div>
		</div>
	</div>

	<!-- Exclusion Filters (collapsible) -->
	<div class="rounded-md border border-gray-200">
		<button
			type="button"
			class="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50"
			onclick={() => (excludeSection = !excludeSection)}
		>
			<span>Exclusion Filters</span>
			<svg
				class="h-4 w-4 transition-transform duration-200"
				class:rotate-180={excludeSection}
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
			</svg>
		</button>

		{#if excludeSection}
			<div class="space-y-4 border-t border-gray-200 px-4 py-4">

				<!-- Exclude Solar -->
				<label class="inline-flex items-center gap-2 text-sm text-gray-700">
					<input type="checkbox" bind:checked={excludeSolar} class="rounded border-gray-300" />
					<span>Exclude solar observations (target / keyword / category contains "sun")</span>
				</label>

				<!-- Exclude keywords & categories -->
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<!-- Exclude Keywords -->
					<div data-dropdown="ex-keyword">
						<span class="text-sm font-medium text-gray-700">Exclude Science Keywords</span>
						{#if excludeSelectedKeywords.length > 0}
							<div class="mt-1 flex flex-wrap gap-1">
								{#each excludeSelectedKeywords as kw}
									<span class="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs text-red-800">
										{kw}
										<button type="button" onclick={() => toggleExcludeKeyword(kw)} class="ml-0.5 text-red-500 hover:text-red-700" aria-label="Remove {kw}">✕</button>
									</span>
								{/each}
								<button type="button" onclick={() => (excludeSelectedKeywords = [])} class="text-xs text-gray-400 hover:text-gray-600 underline">Clear all</button>
							</div>
						{/if}
						<div class="relative mt-1">
							<input
								type="text"
								placeholder={scienceTypes ? 'Search keywords…' : 'Loading…'}
								disabled={!scienceTypes}
								bind:value={excludeKeywordSearch}
								onfocus={() => (excludeKeywordDropdownOpen = true)}
								class="w-full rounded-md border border-gray-300 p-2 text-sm disabled:bg-gray-50"
							/>
							{#if excludeKeywordDropdownOpen && scienceTypes}
								<div class="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
									{#if filteredExcludeKeywords.length === 0}
										<p class="px-3 py-2 text-sm text-gray-400">No results</p>
									{:else}
										{#each filteredExcludeKeywords as kw}
											<label class="flex cursor-pointer items-center gap-2 px-3 py-1.5 hover:bg-gray-50 text-sm">
												<input type="checkbox" checked={excludeSelectedKeywords.includes(kw)} onchange={() => toggleExcludeKeyword(kw)} class="rounded border-gray-300" />
												{kw}
											</label>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
						<p class="mt-0.5 text-xs text-gray-400">{excludeSelectedKeywords.length} excluded</p>
					</div>

					<!-- Exclude Categories -->
					<div data-dropdown="ex-category">
						<span class="text-sm font-medium text-gray-700">Exclude Science Categories</span>
						{#if excludeSelectedCategories.length > 0}
							<div class="mt-1 flex flex-wrap gap-1">
								{#each excludeSelectedCategories as cat}
									<span class="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs text-red-800">
										{cat}
										<button type="button" onclick={() => toggleExcludeCategory(cat)} class="ml-0.5 text-red-500 hover:text-red-700" aria-label="Remove {cat}">✕</button>
									</span>
								{/each}
								<button type="button" onclick={() => (excludeSelectedCategories = [])} class="text-xs text-gray-400 hover:text-gray-600 underline">Clear all</button>
							</div>
						{/if}
						<div class="relative mt-1">
							<input
								type="text"
								placeholder={scienceTypes ? 'Search categories…' : 'Loading…'}
								disabled={!scienceTypes}
								bind:value={excludeCategorySearch}
								onfocus={() => (excludeCategoryDropdownOpen = true)}
								class="w-full rounded-md border border-gray-300 p-2 text-sm disabled:bg-gray-50"
							/>
							{#if excludeCategoryDropdownOpen && scienceTypes}
								<div class="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
									{#if filteredExcludeCategories.length === 0}
										<p class="px-3 py-2 text-sm text-gray-400">No results</p>
									{:else}
										{#each filteredExcludeCategories as cat}
											<label class="flex cursor-pointer items-center gap-2 px-3 py-1.5 hover:bg-gray-50 text-sm">
												<input type="checkbox" checked={excludeSelectedCategories.includes(cat)} onchange={() => toggleExcludeCategory(cat)} class="rounded border-gray-300" />
												{cat}
											</label>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
						<p class="mt-0.5 text-xs text-gray-400">{excludeSelectedCategories.length} excluded</p>
					</div>
				</div>

				<!-- Exclude source name & obs type -->
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label class="block">
							<span class="text-sm font-medium text-gray-700">Exclude Source Names</span>
							<input
								type="text"
								name="exclude_source_name"
								placeholder="e.g. Sun, Moon (comma-separated)"
								class="mt-1 w-full rounded-md border border-gray-300 p-2"
							/>
						</label>
					</div>
					<div>
						<label class="block">
							<span class="text-sm font-medium text-gray-700">Exclude Types</span>
							<input
								type="text"
								name="exclude_obs_type"
								placeholder="e.g. TP (comma-separated)"
								class="mt-1 w-full rounded-md border border-gray-300 p-2"
							/>
						</label>
					</div>
				</div>
			</div>
		{/if}
	</div>
</form>
