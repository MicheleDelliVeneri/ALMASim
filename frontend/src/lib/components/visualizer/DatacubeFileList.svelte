<script lang="ts">
	import { formatBytes, formatDate } from '#lib/utils/format.js';
	interface DatacubeFile {
		name: string;
		path: string;
		size: number;
		modified: number;
		type?: string;
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
		title = 'Available Products',
		emptyMessage = 'No .npz, FITS, or MSv2 files found in the selected directory',
		actionLabel = 'Load'
	}: Props = $props();
</script>

<div class="space-y-4">
	<!-- Output Directory Info -->
	<div
		class="flex items-center gap-2 rounded-md border border-accent-line bg-accent-soft px-3 py-2"
	>
		<span class="shrink-0 text-sm font-medium text-accent">Directory:</span>
		<span class="flex-1 truncate font-mono text-xs text-accent"
			>{outputDir || '(none selected)'}</span
		>
		<button
			type="button"
			onclick={onBrowseRequest}
			class="shrink-0 rounded-md border border-accent-line bg-surface px-2 py-1 text-xs font-medium text-accent hover:bg-accent-soft"
		>
			Browse…
		</button>
	</div>

	<!-- Available Files List -->
	<div class="space-y-2">
		<div class="flex items-center justify-between">
			<span class="block text-sm font-medium text-fg-muted">{title}</span>
			<button
				onclick={onRefresh}
				class="rounded-md bg-surface px-2 py-1 text-xs transition-colors hover:bg-gray-200"
				{disabled}
			>
				{loading ? 'Loading...' : 'Refresh'}
			</button>
		</div>
		{#if !loading && files.length > 0}
			<div class="max-h-64 overflow-y-auto rounded-md border border-line">
				<table class="min-w-full divide-y divide-line">
					<thead class="sticky top-0 bg-surface">
						<tr>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
							>
								File Name
							</th>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
							>
								Size
							</th>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
							>
								Type
							</th>
							<th
								class="px-4 py-2 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
							>
								Modified
							</th>
							<th
								class="px-4 py-2 text-right text-xs font-medium uppercase tracking-wider text-fg-dim"
							>
								Action
							</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-line bg-surface">
						{#each files as file (file.path)}
							<tr class="hover:bg-hover">
								<td class="px-4 py-2 font-mono text-sm text-fg">
									{file.name}
								</td>
								<td class="px-4 py-2 text-sm text-fg-muted">
									{formatBytes(file.size)}
								</td>
								<td class="px-4 py-2 text-sm text-fg-muted uppercase">
									{file.type ?? file.name.split('.').pop() ?? 'file'}
								</td>
								<td class="px-4 py-2 text-sm text-fg-muted">
									{formatDate(file.modified)}
								</td>
								<td class="px-4 py-2 text-right">
									<button
										onclick={() => onFileSelect(file.path)}
										{disabled}
										class="rounded-md bg-primary px-3 py-1 text-xs text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50"
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
			<div class="rounded-md border border-line py-4 text-center text-sm text-fg-dim">
				{loading ? 'Loading files...' : emptyMessage}
			</div>
		{/if}
	</div>
</div>
