<script lang="ts">
	import type { QueryPreset } from '$lib/api/metadata';

	interface Props {
		open: boolean;
		loading: boolean;
		defaultPath: string;
		onClose: () => void;
		onSubmit: (event: Event) => void;
		formRef?: HTMLFormElement;
		presets?: QueryPreset[];
		presetsLoading?: boolean;
		onApplyPreset?: (preset: QueryPreset) => void;
	}

	let {
		open,
		loading,
		defaultPath,
		onClose,
		onSubmit,
		formRef = $bindable(),
		presets = [],
		presetsLoading = false,
		onApplyPreset,
	}: Props = $props();

	type Tab = 'file' | 'preset';
	let activeTab = $state<Tab>('file');
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
			<header class="flex items-center justify-between border-b px-6 py-4">
				<h2 class="text-lg font-semibold text-gray-900">Load Metadata</h2>
				<button
					type="button"
					class="text-gray-500 hover:text-gray-800"
					aria-label="Close"
					onclick={onClose}
				>
					✕
				</button>
			</header>

			<div class="border-b px-6 pt-3">
				<nav class="-mb-px flex gap-4 text-sm" aria-label="Load source">
					<button
						type="button"
						class="border-b-2 px-1 pb-2 font-medium {activeTab === 'file'
							? 'border-gray-900 text-gray-900'
							: 'border-transparent text-gray-500 hover:text-gray-700'}"
						onclick={() => (activeTab = 'file')}
					>
						From file
					</button>
					<button
						type="button"
						class="border-b-2 px-1 pb-2 font-medium {activeTab === 'preset'
							? 'border-gray-900 text-gray-900'
							: 'border-transparent text-gray-500 hover:text-gray-700'}"
						onclick={() => (activeTab = 'preset')}
					>
						From preset
						{#if presets.length > 0}
							<span
								class="ml-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
							>
								{presets.length}
							</span>
						{/if}
					</button>
				</nav>
			</div>

			{#if activeTab === 'file'}
				<form bind:this={formRef} onsubmit={onSubmit} class="space-y-4 px-6 py-6">
					<div class="space-y-1">
						<label class="block text-sm font-medium text-gray-700" for="file_upload">
							Select local file (JSON or CSV)
						</label>
						<input
							id="file_upload"
							name="file_upload"
							type="file"
							accept=".json,.csv,application/json,text/csv"
							class="w-full rounded-md border border-gray-300 p-2 text-sm text-gray-700 file:mr-4 file:rounded file:border-0 file:bg-gray-100 file:px-4 file:py-2 file:text-sm file:font-medium file:text-gray-700 hover:file:bg-gray-200"
						/>
						<p class="text-xs text-gray-500">
							The file contents stay on your machine; we parse supported formats directly in
							the browser.
						</p>
					</div>

					<div class="space-y-1">
						<label class="block text-sm font-medium text-gray-700" for="file_path">
							Or load from server path (JSON or CSV)
						</label>
						<input
							type="text"
							id="file_path"
							name="file_path"
							placeholder="data/metadata/sample.json"
							value={defaultPath}
							class="w-full rounded-md border border-gray-300 p-2"
						/>
						<p class="text-xs text-gray-500">
							Use when the backend already has access to the metadata file.
						</p>
					</div>

					<div class="flex items-center justify-end gap-2">
						<button
							type="button"
							class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
							onclick={onClose}
						>
							Cancel
						</button>
						<button
							type="submit"
							disabled={loading}
							class="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800 disabled:bg-gray-600"
						>
							{loading ? 'Loading...' : 'Load'}
						</button>
					</div>
				</form>
			{:else}
				<div class="space-y-4 px-6 py-6">
					{#if presetsLoading}
						<p class="text-sm text-gray-500">Loading presets…</p>
					{:else if presets.length === 0}
						<p class="text-sm text-gray-500">
							No saved presets yet. Run a query and save it as a preset to reuse it later.
						</p>
					{:else}
						<p class="text-sm text-gray-600">
							Pick a preset to re-run its query and load the matching metadata.
						</p>
						<ul class="max-h-72 space-y-2 overflow-y-auto pr-1">
							{#each presets as preset (preset.name)}
								<li
									class="flex items-start justify-between gap-3 rounded-md border border-gray-200 px-3 py-2"
								>
									<div class="min-w-0">
										<p class="truncate text-sm font-medium text-gray-900">
											{preset.name}
										</p>
										{#if preset.description}
											<p class="truncate text-xs text-gray-500">
												{preset.description}
											</p>
										{/if}
										<p class="text-xs text-gray-400">
											{preset.result_count} rows · saved
											{preset.created_at?.slice(0, 10) ?? ''}
										</p>
									</div>
									<button
										type="button"
										class="shrink-0 rounded-md bg-gray-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-gray-800 disabled:bg-gray-600"
										disabled={loading || !onApplyPreset}
										onclick={() => onApplyPreset?.(preset)}
									>
										Run
									</button>
								</li>
							{/each}
						</ul>
					{/if}

					<div class="flex items-center justify-end gap-2">
						<button
							type="button"
							class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
							onclick={onClose}
						>
							Close
						</button>
					</div>
				</div>
			{/if}
		</div>
	</div>
{/if}
