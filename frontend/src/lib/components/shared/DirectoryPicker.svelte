<script lang="ts">
	import { onMount } from 'svelte';
	import { downloadApi, type BrowseDirectoryResponse } from '#lib/api/download.js';
	import { createLogger } from '#lib/logger.js';
	import { childPath } from '#lib/utils/paths.js';
	import Dialog from './Dialog.svelte';
	import Spinner from './Spinner.svelte';

	const logger = createLogger('components/shared/DirectoryPicker');

	interface Props {
		/** Directory shown first. */
		startPath: string;
		onSelect: (path: string) => void;
		onClose: () => void;
		title?: string;
		selectLabel?: string;
		/** Offer a "new folder" row in the current directory. */
		allowCreate?: boolean;
	}

	let {
		startPath,
		onSelect,
		onClose,
		title = 'Select Directory',
		selectLabel = 'Select this folder',
		allowCreate = false
	}: Props = $props();

	let result = $state<BrowseDirectoryResponse | null>(null);
	let browsing = $state(false);
	let error = $state('');
	let newFolderName = $state('');
	let creating = $state(false);

	async function browse(path: string) {
		browsing = true;
		error = '';
		try {
			result = await downloadApi.browseDirectory(path);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to browse directory';
			logger.error({ path, err: e }, 'Failed to browse directory');
		} finally {
			browsing = false;
		}
	}

	async function createFolder() {
		if (!result) return;
		const path = childPath(result.current, newFolderName);
		if (!path) return;
		creating = true;
		error = '';
		try {
			result = await downloadApi.createDirectory(path);
			newFolderName = '';
			logger.info({ path }, 'Folder created');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to create folder';
			logger.error({ path, err: e }, 'Failed to create folder');
		} finally {
			creating = false;
		}
	}

	function select() {
		if (!result) return;
		logger.info({ path: result.current }, 'Directory selected');
		onSelect(result.current);
	}

	onMount(() => {
		void browse(startPath);
	});
</script>

<Dialog {title} {onClose}>
	<div class="space-y-3">
		{#if error}
			<div class="rounded-md bg-danger-soft px-4 py-3 text-sm text-danger">{error}</div>
		{/if}

		{#if browsing && !result}
			<div class="flex items-center justify-center gap-2 py-6 text-sm text-fg-dim">
				<Spinner />
				Loading…
			</div>
		{:else if result}
			{@const current = result}
			<div class="flex items-center gap-2 rounded-md bg-sunk px-3 py-2">
				<span class="truncate font-mono text-xs text-fg-muted" title={current.current}>
					{current.current}
				</span>
				{#if browsing}
					<Spinner class="h-3.5 w-3.5 shrink-0 text-fg-dim" />
				{/if}
			</div>

			<ul class="max-h-64 overflow-y-auto rounded-md border border-line bg-sunk">
				{#if current.parent}
					<li class="border-b border-line">
						<button
							type="button"
							class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-hover"
							onclick={() => browse(current.parent!)}
						>
							<span class="text-fg-dim">↩</span>
							<span class="text-fg-dim">..</span>
						</button>
					</li>
				{/if}
				{#each current.entries as entry (entry.path)}
					<li class="border-b border-line last:border-b-0">
						<button
							type="button"
							class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-accent-soft"
							onclick={() => browse(entry.path)}
						>
							<span class="text-yellow-500">📁</span>
							<span class="truncate text-fg-muted">{entry.name}</span>
						</button>
					</li>
				{:else}
					<li class="px-3 py-4 text-center text-sm italic text-fg-dim">No subdirectories</li>
				{/each}
			</ul>

			{#if allowCreate}
				<div class="flex gap-2">
					<input
						type="text"
						bind:value={newFolderName}
						placeholder="New folder name…"
						aria-label="New folder name"
						class="flex-1 rounded-md border border-line px-3 py-1.5 text-sm"
						onkeydown={(e) => e.key === 'Enter' && createFolder()}
					/>
					<button
						type="button"
						disabled={!childPath(current.current, newFolderName) || creating}
						onclick={createFolder}
						class="rounded-md border border-line px-3 py-1.5 text-sm font-medium text-fg-muted hover:bg-hover disabled:opacity-50"
					>
						{creating ? '…' : '+ Create'}
					</button>
				</div>
			{/if}
		{/if}
	</div>

	{#snippet footer()}
		<button
			type="button"
			class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover"
			onclick={onClose}
		>
			Cancel
		</button>
		<button
			type="button"
			disabled={!result || browsing}
			class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50"
			onclick={select}
		>
			{selectLabel}
		</button>
	{/snippet}
</Dialog>
