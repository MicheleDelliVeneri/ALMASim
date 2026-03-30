<script lang="ts">
	interface DatacubeFile {
		name: string;
		path: string;
		size: number;
		modified: number;
	}

	interface Props {
		files: DatacubeFile[];
		loading: boolean;
		outputDir: string;
		onFileSelect: (path: string) => void;
		onRefresh: () => void;
		onBrowseRequest: () => void;
		disabled: boolean;
		title?: string;
		emptyMessage?: string;
		actionLabel?: string;
	}

	let {
		files,
		loading,
		outputDir,
		onFileSelect,
		onRefresh,
		onBrowseRequest,
		disabled,
		title = 'Available Datacubes',
		emptyMessage = 'No .npz files found in output directory',
		actionLabel = 'Load'
	}: Props = $props();

	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatDate(timestamp: number): string {
		return new Date(timestamp * 1000).toLocaleString();
	}
</script>

<div class="space-y-4">
	<!-- Output Directory Info -->
	<div class="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2">
		<span class="shrink-0 text-sm font-medium text-blue-800">Directory:</span>
		<span class="flex-1 truncate font-mono text-xs text-blue-700">{outputDir || '(none selected)'}</span>
		<button
			type="button"
			onclick={onBrowseRequest}
			class="shrink-0 rounded-md border border-blue-300 bg-white px-2 py-1 text-xs font-medium text-blue-700 hover:bg-blue-50"
		>
			Browse…
		</button>
	</div>

	<!-- Available Files List -->
	<div class="space-y-2">
		<div class="flex items-center justify-between">
			<span class="block text-sm font-medium text-gray-700">{title}</span>
			<button
				onclick={onRefresh}
				class="rounded-md bg-gray-100 px-2 py-1 text-xs transition-colors hover:bg-gray-200"
				{disabled}
			>
				{loading ? 'Loading...' : 'Refresh'}
			</button>
		</div>
		{#if !loading && files.length > 0}
			<div class="max-h-64 overflow-y-auto rounded-md border border-gray-200">
				<table class="min-w-full divide-y divide-gray-200">
					<thead class="sticky top-0 bg-gray-50">
						<tr>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500"
							>
								File Name
							</th>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500"
							>
								Size
							</th>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-gray-500"
							>
								Modified
							</th>
							<th
								class="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-gray-500"
							>
								Action
							</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-gray-200 bg-white">
						{#each files as file}
							<tr class="hover:bg-gray-50">
								<td class="px-4 py-2 font-mono text-sm text-gray-900">
									{file.name}
								</td>
								<td class="px-4 py-2 text-sm text-gray-600">
									{formatFileSize(file.size)}
								</td>
								<td class="px-4 py-2 text-sm text-gray-600">
									{formatDate(file.modified)}
								</td>
								<td class="px-4 py-2 text-right">
									<button
										onclick={() => onFileSelect(file.path)}
										disabled={disabled}
										class="rounded-md bg-blue-600 px-3 py-1 text-xs text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
									>
										{actionLabel}
									</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<div class="rounded-md border border-gray-200 py-4 text-center text-sm text-gray-500">
				{loading ? 'Loading files...' : emptyMessage}
			</div>
		{/if}
	</div>
</div>
