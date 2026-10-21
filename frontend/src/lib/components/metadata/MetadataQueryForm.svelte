<script lang="ts">
	import type { MetadataQuery } from '#lib/api/metadata.js';
	import { createLogger } from '#lib/logger.js';
	import MultiSelect from '#lib/components/shared/MultiSelect.svelte';

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
		label: `Cycle ${i} (${2012 + i})`
	}));
	let selectedCycles = $state<number[]>([]);
	const qa2Options = [
		{ value: 'Pass', label: 'Pass' },
		{ value: 'SemiPass', label: 'Semi-Pass' },
		{ value: 'Fail', label: 'Fail' }
	];

	// Inclusion multi-select state
	let selectedKeywords = $state<string[]>([]);
	let selectedCategories = $state<string[]>([]);

	// Exclusion state
	let excludeSection = $state(false);
	let excludeSelectedKeywords = $state<string[]>([]);
	let excludeSelectedCategories = $state<string[]>([]);
	let excludeSolar = $state(false);
	let publicOnly = $state(true);
	let scienceOnly = $state(true);
	let excludeMosaic = $state(true);

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
			query.array_configuration = arrayConfig
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
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
		if (excludeSelectedCategories.length)
			query.exclude_scientific_category = excludeSelectedCategories;

		const excludeSourceName = formData.get('exclude_source_name') as string;
		if (excludeSourceName?.trim()) {
			query.exclude_source_name = excludeSourceName
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
		}

		const excludeObsType = formData.get('exclude_obs_type') as string;
		if (excludeObsType?.trim()) {
			query.exclude_obs_type = excludeObsType
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean);
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
				scienceOnly
			},
			'Query form submitted'
		);
		onSubmit(query);
	}
</script>

<form onsubmit={handleSubmit} class="space-y-4 rounded-lg bg-surface p-6 shadow">
	<div class="flex items-center justify-between">
		<h2 class="text-xl font-semibold text-fg">Query Builder</h2>
	</div>

	<!-- Quick-filter toggles -->
	<div class="flex flex-wrap gap-x-6 gap-y-2">
		<label class="inline-flex items-center gap-2 text-sm text-fg-muted">
			<input type="checkbox" bind:checked={publicOnly} class="rounded border-line" />
			<span>Public data only</span>
		</label>
	</div>

	<!-- Source Name -->
	<div>
		<label class="block">
			<span class="text-sm font-medium text-fg-muted">Source Name</span>
			<input
				type="text"
				name="source_name"
				placeholder="e.g. NGC 1068"
				class="mt-1 w-full rounded-md border border-line p-2"
			/>
		</label>
	</div>

	<!-- Science Keywords & Categories -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<!-- Science Keywords multi-select -->
		<MultiSelect
			label="Science Keywords"
			options={scienceTypes?.keywords ?? null}
			bind:selected={selectedKeywords}
			placeholder="Search keywords…"
		/>

		<!-- Science Categories multi-select -->
		<MultiSelect
			label="Science Category"
			options={scienceTypes?.categories ?? null}
			bind:selected={selectedCategories}
			placeholder="Search categories…"
		/>
	</div>

	<!-- ALMA Cycle -->
	<div>
		<div class="flex items-center justify-between">
			<span class="text-sm font-medium text-fg-muted">ALMA Cycle</span>
			{#if selectedCycles.length > 0}
				<button
					type="button"
					onclick={() => (selectedCycles = [])}
					class="text-xs text-fg-dim hover:text-gray-600 underline">Clear all</button
				>
			{/if}
		</div>
		<div class="mt-2 flex flex-wrap gap-2">
			{#each cycleOptions as opt (opt.cycle)}
				<label class="inline-flex items-center space-x-1 text-sm text-fg-muted">
					<input
						type="checkbox"
						checked={selectedCycles.includes(opt.cycle)}
						onchange={() => {
							selectedCycles = selectedCycles.includes(opt.cycle)
								? selectedCycles.filter((c) => c !== opt.cycle)
								: [...selectedCycles, opt.cycle];
						}}
						class="rounded border-line"
					/>
					<span>{opt.label}</span>
				</label>
			{/each}
		</div>
		<p class="mt-0.5 text-xs text-fg-dim">
			{selectedCycles.length > 0
				? `Filtering by ${selectedCycles.length} cycle(s): proposal_id prefix ${selectedCycles.map((c) => `${2012 + c}.`).join(', ')}`
				: 'All cycles (no filter)'}
		</p>
	</div>

	<!-- Bands -->
	<div>
		<span class="text-sm font-medium text-fg-muted">Band</span>
		<div class="mt-2 flex flex-wrap gap-2">
			{#each bandOptions as band (band)}
				<label class="inline-flex items-center space-x-1 text-sm text-fg-muted">
					<input type="checkbox" name="bands" value={band} class="rounded border-line" />
					<span>Band {band}</span>
				</label>
			{/each}
		</div>
	</div>

	<!-- Array Type & Array Configuration -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="text-sm font-medium text-fg-muted">Array Type</span>
			<div class="mt-2 flex gap-4">
				{#each arrayTypeOptions as atype (atype)}
					<label class="inline-flex items-center space-x-1 text-sm text-fg-muted">
						<input type="checkbox" name="array_type" value={atype} class="rounded border-line" />
						<span>{atype}</span>
					</label>
				{/each}
			</div>
		</div>
		<div>
			<label class="block">
				<span class="text-sm font-medium text-fg-muted">Array Configuration</span>
				<input
					type="text"
					name="array_configuration"
					placeholder="e.g. C-1, C-2 (comma-separated)"
					class="mt-1 w-full rounded-md border border-line p-2"
				/>
			</label>
		</div>
	</div>

	<!-- Angular Resolution & Observation Date -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="block text-sm font-medium text-fg-muted">Angular Resolution (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.01"
					name="ang_res_min"
					placeholder="Min"
					class="w-full rounded-md border border-line p-2"
				/>
				<input
					type="number"
					step="0.01"
					name="ang_res_max"
					placeholder="Max"
					class="w-full rounded-md border border-line p-2"
				/>
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-fg-muted">Observation Date</span>
			<div class="mt-1 flex gap-2">
				<input type="date" name="obs_date_min" class="w-full rounded-md border border-line p-2" />
				<input type="date" name="obs_date_max" class="w-full rounded-md border border-line p-2" />
			</div>
		</div>
	</div>

	<!-- QA2 Status & Type -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		<div>
			<span class="text-sm font-medium text-fg-muted">QA2 Status</span>
			<div class="mt-2 flex gap-4">
				{#each qa2Options as qa2 (qa2.value)}
					<label class="inline-flex items-center space-x-1 text-sm text-fg-muted">
						<input
							type="checkbox"
							name="qa2_status"
							value={qa2.value}
							class="rounded border-line"
						/>
						<span>{qa2.label}</span>
					</label>
				{/each}
			</div>
		</div>
		<div>
			<div>
				<span class="text-sm font-medium text-fg-muted">Project Type</span>
				<div class="mt-2 flex flex-wrap gap-3">
					{#each [['S', 'Science'], ['L', 'Large'], ['SV', 'Science Verification'], ['V', 'VLBI'], ['T', 'Target of Opportunity'], ['P', 'Phased Array']] as [val, lbl] (val)}
						<label class="inline-flex items-center space-x-1 text-sm text-fg-muted">
							<input type="checkbox" name="obs_type" value={val} class="rounded border-line" />
							<span>{val} ({lbl})</span>
						</label>
					{/each}
				</div>
			</div>
			<div class="mt-2 flex gap-4">
				<label class="inline-flex items-center gap-1.5 text-sm text-fg-muted">
					<input type="checkbox" bind:checked={scienceOnly} class="rounded border-line" />
					<span>Science only</span>
				</label>
				<label class="inline-flex items-center gap-1.5 text-sm text-fg-muted">
					<input type="checkbox" bind:checked={excludeMosaic} class="rounded border-line" />
					<span>Exclude mosaics</span>
				</label>
			</div>
		</div>
	</div>

	<!-- FOV, Time Resolution, Frequency -->
	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<span class="block text-sm font-medium text-fg-muted">FOV (arcsec)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="fov_min"
					placeholder="Min"
					class="w-full rounded-md border border-line p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="fov_max"
					placeholder="Max"
					class="w-full rounded-md border border-line p-2"
				/>
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-fg-muted">Time Resolution (s)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="time_min"
					placeholder="Min"
					class="w-full rounded-md border border-line p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="time_max"
					placeholder="Max"
					class="w-full rounded-md border border-line p-2"
				/>
			</div>
		</div>
		<div>
			<span class="block text-sm font-medium text-fg-muted">Frequency (GHz)</span>
			<div class="mt-1 flex gap-2">
				<input
					type="number"
					step="0.1"
					name="freq_min"
					placeholder="Min"
					class="w-full rounded-md border border-line p-2"
				/>
				<input
					type="number"
					step="0.1"
					name="freq_max"
					placeholder="Max"
					class="w-full rounded-md border border-line p-2"
				/>
			</div>
		</div>
	</div>

	<!-- Exclusion Filters (collapsible) -->
	<div class="rounded-md border border-line">
		<button
			type="button"
			class="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-fg-muted hover:bg-hover"
			onclick={() => (excludeSection = !excludeSection)}
		>
			<span>Exclusion Filters</span>
			<svg
				class="h-4 w-4 transition-transform duration-200"
				class:rotate-180={excludeSection}
				fill="none"
				stroke="currentColor"
				viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
			</svg>
		</button>

		{#if excludeSection}
			<div class="space-y-4 border-t border-line px-4 py-4">
				<!-- Exclude Solar -->
				<label class="inline-flex items-center gap-2 text-sm text-fg-muted">
					<input type="checkbox" bind:checked={excludeSolar} class="rounded border-line" />
					<span>Exclude solar observations (target / keyword / category contains "sun")</span>
				</label>

				<!-- Exclude keywords & categories -->
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<!-- Exclude Keywords -->
					<MultiSelect
						label="Exclude Science Keywords"
						options={scienceTypes?.keywords ?? null}
						bind:selected={excludeSelectedKeywords}
						placeholder="Search keywords…"
						tone="danger"
					/>

					<!-- Exclude Categories -->
					<MultiSelect
						label="Exclude Science Categories"
						options={scienceTypes?.categories ?? null}
						bind:selected={excludeSelectedCategories}
						placeholder="Search categories…"
						tone="danger"
					/>
				</div>

				<!-- Exclude source name & obs type -->
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label class="block">
							<span class="text-sm font-medium text-fg-muted">Exclude Source Names</span>
							<input
								type="text"
								name="exclude_source_name"
								placeholder="e.g. Sun, Moon (comma-separated)"
								class="mt-1 w-full rounded-md border border-line p-2"
							/>
						</label>
					</div>
					<div>
						<label class="block">
							<span class="text-sm font-medium text-fg-muted">Exclude Types</span>
							<input
								type="text"
								name="exclude_obs_type"
								placeholder="e.g. TP (comma-separated)"
								class="mt-1 w-full rounded-md border border-line p-2"
							/>
						</label>
					</div>
				</div>
			</div>
		{/if}
	</div>

	<div class="flex justify-end pt-2">
		<button
			type="submit"
			disabled={loading}
			class="rounded-md bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:bg-gray-400"
		>
			{loading ? 'Running...' : 'Run Query'}
		</button>
	</div>
</form>
