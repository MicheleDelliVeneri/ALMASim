<script lang="ts">
	// A searchable multi-select with removable chips (MADIV-28 T7.5): the query
	// form had four copies of it, one per keyword/category list.
	interface Props {
		label: string;
		/** The choices; null while they are loading. */
		options: string[] | null;
		selected: string[];
		placeholder?: string;
		tone?: 'accent' | 'danger';
	}

	let {
		label,
		options,
		selected = $bindable([]),
		placeholder = 'Search…',
		tone = 'accent'
	}: Props = $props();

	let search = $state('');
	let open = $state(false);
	let root: HTMLDivElement | undefined = $state();

	const filtered = $derived(
		(options ?? []).filter((option) => option.toLowerCase().includes(search.toLowerCase()))
	);
	const chipClass = $derived(
		tone === 'danger' ? 'bg-danger-soft text-danger' : 'bg-accent-soft text-accent'
	);
	const removeClass = $derived(
		tone === 'danger' ? 'text-danger hover:text-danger' : 'text-accent hover:text-accent'
	);

	function toggle(option: string) {
		selected = selected.includes(option)
			? selected.filter((value) => value !== option)
			: [...selected, option];
	}
</script>

<svelte:window
	onclick={(e) => {
		if (root && !root.contains(e.target as Node)) open = false;
	}}
/>

<div bind:this={root}>
	<span class="text-sm font-medium text-fg-muted">{label}</span>
	<div class="relative mt-1">
		<input
			type="text"
			placeholder={options ? placeholder : 'Loading…'}
			disabled={!options}
			bind:value={search}
			onfocus={() => (open = true)}
			class="w-full rounded-md border border-line p-2 text-sm disabled:bg-gray-50"
		/>
		{#if open && options}
			<div
				class="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-line bg-surface shadow-panel"
			>
				{#if filtered.length === 0}
					<p class="px-3 py-2 text-sm text-fg-dim">No results</p>
				{:else}
					{#each filtered as option (option)}
						<label
							class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-hover"
						>
							<input
								type="checkbox"
								checked={selected.includes(option)}
								onchange={() => toggle(option)}
								class="rounded border-line"
							/>
							{option}
						</label>
					{/each}
				{/if}
			</div>
		{/if}
	</div>
	{#if selected.length > 0}
		<div class="mt-1 flex flex-wrap gap-1">
			{#each selected as option (option)}
				<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs {chipClass}">
					{option}
					<button
						type="button"
						onclick={() => toggle(option)}
						class="ml-0.5 {removeClass}"
						aria-label="Remove {option}">✕</button
					>
				</span>
			{/each}
			<button
				type="button"
				onclick={() => (selected = [])}
				class="text-xs text-fg-dim underline hover:text-gray-600">Clear all</button
			>
		</div>
	{:else}
		<p class="mt-0.5 text-xs text-fg-dim">0 selected</p>
	{/if}
</div>
