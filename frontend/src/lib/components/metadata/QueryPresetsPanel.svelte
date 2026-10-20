<script lang="ts">
	import type { QueryPreset } from '#lib/api/metadata.js';

	interface Props {
		presets: QueryPreset[];
		onApply: (preset: QueryPreset) => void;
		onDelete: (name: string) => void;
	}

	let { presets, onApply, onDelete }: Props = $props();
</script>

{#if presets.length > 0}
	<div class="rounded-lg border border-indigo-100 bg-accent-soft p-4">
		<h2 class="mb-3 text-sm font-semibold text-accent">Saved Query Presets</h2>
		<div class="space-y-2">
			{#each presets as preset (preset.name)}
				<div
					class="flex items-center justify-between rounded-md border border-accent-line bg-surface px-3 py-2 text-sm"
				>
					<div class="min-w-0">
						<span class="font-medium text-fg">{preset.name}</span>
						{#if preset.description}
							<span class="ml-2 text-xs text-fg-dim">{preset.description}</span>
						{/if}
						<span class="ml-2 text-xs text-fg-dim">({preset.result_count} rows)</span>
					</div>
					<div class="ml-4 flex shrink-0 gap-2">
						<button
							onclick={() => onApply(preset)}
							class="rounded-md bg-primary px-2 py-1 text-xs font-medium text-white hover:bg-primary-hover"
							>Apply & Run</button
						>
						<button
							onclick={() => onDelete(preset.name)}
							class="rounded-md bg-surface px-2 py-1 text-xs font-medium text-danger ring-1 ring-red-200 hover:bg-danger-soft"
							>Delete</button
						>
					</div>
				</div>
			{/each}
		</div>
	</div>
{/if}
