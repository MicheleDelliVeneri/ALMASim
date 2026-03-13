<script lang="ts">
	import type { ResolveProductsResponse, DiskSpaceInfo, BrowseDirectoryResponse } from '$lib/api/download';
	import { downloadApi } from '$lib/api/download';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('components/metadata/DownloadDialog');

	interface Props {
		open: boolean;
		memberOusUids: string[];
		onClose: () => void;
		onStarted: (jobId: string) => void;
	}

	let { open, memberOusUids, onClose, onStarted }: Props = $props();

	// State
	let resolving = $state(false);
	let resolved = $state<ResolveProductsResponse | null>(null);
	let resolveError = $state('');

	let productFilter = $state('all');
	let destination = $state('/host_home/Downloads');
	let maxParallel = $state(3);

	let diskSpace = $state<DiskSpaceInfo | null>(null);
	let diskSpaceLoading = $state(false);
	let diskSpaceError = $state('');

	let starting = $state(false);
	let startError = $state('');

	$effect(() => {
		logger.debug({ productFilter }, 'Product filter changed');
	});

	// Directory browser state
	let browserOpen = $state(false);
	let browsing = $state(false);
	let browseResult = $state<BrowseDirectoryResponse | null>(null);
	let browseError = $state('');

	async function browseDir(path: string) {
		browsing = true;
		browseError = '';
		try {
			browseResult = await downloadApi.browseDirectory(path);
		} catch (e) {
			browseError = e instanceof Error ? e.message : 'Failed to browse directory';
		} finally {
			browsing = false;
		}
	}

	function openBrowser() {
		browserOpen = true;
		browseDir(destination || '/');
	}

	function selectDirectory(path: string) {
		logger.debug({ path }, 'Download destination selected');
		destination = path;
		browserOpen = false;
		browseResult = null;
	}

	// Filtered size/count
	const filteredInfo = $derived.by(() => {
		if (!resolved) return { count: 0, size: 0, display: '0 B' };
		if (productFilter === 'all') {
			return {
				count: resolved.total_count,
				size: resolved.total_size_bytes,
				display: resolved.total_size_display
			};
		}
		const info = resolved.by_type[productFilter];
		if (!info) return { count: 0, size: 0, display: '0 B' };
		return { count: info.count, size: info.size_bytes, display: info.size_display };
	});

	// Resolve products when dialog opens
	$effect(() => {
		if (open && memberOusUids.length > 0 && !resolved) {
			resolveProducts();
		}
	});

	// Re-check disk space when destination or filter changes
	$effect(() => {
		if (open && destination && filteredInfo.size > 0) {
			checkDisk();
		}
	});

	// Reset state when closing
	$effect(() => {
		if (!open) {
			resolved = null;
			resolveError = '';
			diskSpace = null;
			diskSpaceError = '';
			startError = '';
			productFilter = 'all';
			browserOpen = false;
			browseResult = null;
			browseError = '';
		}
	});

	async function resolveProducts() {
		logger.info({ count: memberOusUids.length }, 'Resolving ALMA products');
		resolving = true;
		resolveError = '';
		try {
			resolved = await downloadApi.resolveProducts(memberOusUids);
			logger.info({ totalCount: resolved.total_count, totalSize: resolved.total_size_display }, 'Products resolved');
		} catch (e) {
			resolveError = e instanceof Error ? e.message : 'Failed to resolve products';
			logger.error({ err: e }, 'Failed to resolve products');
		} finally {
			resolving = false;
		}
	}

	async function checkDisk() {
		diskSpaceLoading = true;
		diskSpaceError = '';
		try {
			diskSpace = await downloadApi.checkDiskSpace(destination, filteredInfo.size);
		} catch (e) {
			diskSpaceError = e instanceof Error ? e.message : 'Failed to check disk space';
			diskSpace = null;
		} finally {
			diskSpaceLoading = false;
		}
	}

	async function startDownload() {
		logger.info({ productFilter, destination, maxParallel, fileCount: filteredInfo.count }, 'Starting download');
		starting = true;
		startError = '';
		try {
			const resp = await downloadApi.startDownload({
				memberOusUids,
				productFilter,
				destination,
				maxParallel
			});
			logger.info({ jobId: resp.job_id }, 'Download job created');
			onStarted(resp.job_id);
			onClose();
		} catch (e) {
			startError = e instanceof Error ? e.message : 'Failed to start download';
			logger.error({ err: e }, 'Failed to start download');
		} finally {
			starting = false;
		}
	}

	const productTypes = [
		{ value: 'all', label: 'All Products' },
		{ value: 'fits', label: 'FITS' },
		{ value: 'raw', label: 'Raw (ASDM)' },
		{ value: 'calibration', label: 'Calibration' },
		{ value: 'scripts', label: 'Pipeline Scripts' },
		{ value: 'weblog', label: 'Web Logs' },
		{ value: 'qa_reports', label: 'QA Reports' },
		{ value: 'cubes', label: 'Spectral Cubes' },
		{ value: 'continuum', label: 'Continuum Images' },
		{ value: 'auxiliary', label: 'Auxiliary' },
		{ value: 'other', label: 'Other' }
	];
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-xl rounded-lg bg-white shadow-xl">
			<header class="flex items-center justify-between border-b px-6 py-4">
				<h2 class="text-lg font-semibold text-gray-900">Download ALMA Data</h2>
				<button
					type="button"
					class="text-gray-500 hover:text-gray-800"
					aria-label="Close"
					onclick={onClose}
				>
					✕
				</button>
			</header>

			<div class="space-y-5 px-6 py-6">
				<!-- Observations summary -->
				<p class="text-sm text-gray-600">
					{memberOusUids.length} observation{memberOusUids.length === 1 ? '' : 's'} selected.
				</p>

				<!-- Loading / Error state -->
				{#if resolving}
					<div class="flex items-center gap-2 text-sm text-gray-500">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
							<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
						</svg>
						Resolving available products…
					</div>
				{:else if resolveError}
					<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
						{resolveError}
						<button type="button" class="ml-2 underline" onclick={resolveProducts}>Retry</button>
					</div>
				{:else if resolved}
					<!-- Product type selector -->
					<div>
						<label class="block text-sm font-medium text-gray-700 mb-2" for="product-filter">
							Product type
						</label>
						<div class="flex flex-wrap gap-2">
							{#each productTypes as pt}
								{@const typeInfo = pt.value === 'all' ? null : resolved.by_type[pt.value]}
								{@const count = pt.value === 'all' ? resolved.total_count : (typeInfo?.count ?? 0)}
								<button
									type="button"
									class="rounded-full px-3 py-1 text-sm border transition-colors"
									class:bg-blue-600={productFilter === pt.value}
									class:text-white={productFilter === pt.value}
									class:border-blue-600={productFilter === pt.value}
									class:bg-white={productFilter !== pt.value}
									class:text-gray-700={productFilter !== pt.value}
									class:border-gray-300={productFilter !== pt.value}
									class:opacity-50={count === 0 && pt.value !== 'all'}
									disabled={count === 0 && pt.value !== 'all'}
									onclick={() => (productFilter = pt.value)}
								>
									{pt.label} ({count})
								</button>
							{/each}
						</div>
					</div>

					<!-- Size info -->
					<div class="rounded-md bg-gray-50 p-3 text-sm text-gray-700">
						<span class="font-medium">{filteredInfo.count}</span> files,
						<span class="font-medium">{filteredInfo.display}</span> total
					</div>

					<!-- Destination path -->
					<div>
						<label class="block text-sm font-medium text-gray-700 mb-1" for="download-dest">
							Download destination
						</label>
						<div class="flex gap-2">
							<input
								type="text"
								id="download-dest"
								bind:value={destination}
								class="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm"
								placeholder="/path/to/download/directory"
							/>
							<button
								type="button"
								class="rounded-md border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 whitespace-nowrap"
								onclick={openBrowser}
							>
								Browse…
							</button>
						</div>

						<!-- Inline directory browser -->
					</div>

					<!-- Disk space -->
					{#if diskSpaceLoading}
						<p class="text-xs text-gray-400">Checking disk space…</p>
					{:else if diskSpaceError}
						<p class="text-xs text-red-500">{diskSpaceError}</p>
					{:else if diskSpace}
						<div
							class="rounded-md p-3 text-sm"
							class:bg-green-50={diskSpace.sufficient}
							class:text-green-700={diskSpace.sufficient}
							class:bg-red-50={!diskSpace.sufficient}
							class:text-red-700={!diskSpace.sufficient}
						>
							Free: <span class="font-medium">{diskSpace.free_display}</span>
							/ Total: {diskSpace.total_display}
							{#if !diskSpace.sufficient}
								<span class="ml-2 font-semibold">— Insufficient space!</span>
							{/if}
						</div>
					{/if}

					<!-- Parallel downloads -->
					<div>
						<label class="block text-sm font-medium text-gray-700 mb-1" for="max-parallel">
							Parallel downloads
						</label>
						<input
							type="range"
							id="max-parallel"
							min="1"
							max="8"
							bind:value={maxParallel}
							class="w-full"
						/>
						<p class="text-xs text-gray-500">{maxParallel} simultaneous files</p>
					</div>

					<!-- Errors -->
					{#if startError}
						<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{startError}</div>
					{/if}
				{/if}

				<!-- Actions -->
				<div class="flex items-center justify-end gap-3 pt-2 border-t">
					<button
						type="button"
						class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
						onclick={onClose}
					>
						Cancel
					</button>
					<button
						type="button"
						disabled={!resolved || filteredInfo.count === 0 || starting || (diskSpace != null && !diskSpace.sufficient)}
						class="rounded-md bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed"
						onclick={startDownload}
					>
						{starting ? 'Starting…' : `Download ${filteredInfo.count} files`}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<!-- Browse directory popup -->
{#if browserOpen}
	<div
		class="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-lg rounded-lg bg-white shadow-2xl">
			<header class="flex items-center justify-between border-b px-5 py-3">
				<h3 class="text-sm font-semibold text-gray-900">Choose Download Folder</h3>
				<button
					type="button"
					class="text-gray-400 hover:text-gray-700"
					aria-label="Close"
					onclick={() => { browserOpen = false; browseResult = null; browseError = ''; }}
				>
					✕
				</button>
			</header>

			<div class="px-5 py-4">
				{#if browsing && !browseResult}
					<div class="flex items-center gap-2 py-6 justify-center text-sm text-gray-500">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
							<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
						</svg>
						Loading…
					</div>
				{:else if browseError}
					<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{browseError}</div>
				{:else if browseResult}
					<!-- Current path breadcrumb -->
					<div class="mb-3 flex items-center gap-2 rounded-md bg-gray-100 px-3 py-2">
						<span class="truncate font-mono text-xs text-gray-600" title={browseResult.current}>
							{browseResult.current}
						</span>
						{#if browsing}
							<svg class="h-3.5 w-3.5 shrink-0 animate-spin text-gray-400" viewBox="0 0 24 24" fill="none">
								<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
								<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
							</svg>
						{/if}
					</div>

					<!-- Directory listing -->
					<ul class="max-h-64 overflow-y-auto rounded-md border border-gray-200 bg-white">
						{#if browseResult.parent}
							<li class="border-b border-gray-100">
								<button
									type="button"
									class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-gray-50"
									onclick={() => browseDir(browseResult!.parent!)}
								>
									<span class="text-gray-400">↩</span>
									<span class="text-gray-500">..</span>
								</button>
							</li>
						{/if}
						{#each browseResult.entries as entry}
							<li class="border-b border-gray-100 last:border-b-0">
								<button
									type="button"
									class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-blue-50"
									onclick={() => browseDir(entry.path)}
								>
									<span class="text-yellow-500">📁</span>
									<span class="truncate text-gray-700">{entry.name}</span>
								</button>
							</li>
						{:else}
							<li class="px-3 py-4 text-center text-sm text-gray-400 italic">No subdirectories</li>
						{/each}
					</ul>
				{/if}
			</div>

			<footer class="flex items-center justify-end gap-3 border-t px-5 py-3">
				<button
					type="button"
					class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
					onclick={() => { browserOpen = false; browseResult = null; browseError = ''; }}
				>
					Cancel
				</button>
				<button
					type="button"
					disabled={!browseResult}
					class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
					onclick={() => browseResult && selectDirectory(browseResult.current)}
				>
					Select this folder
				</button>
			</footer>
		</div>
	</div>
{/if}
