<script lang="ts">
	import type { MetadataRow } from '#lib/api/metadata.js';
	import type { ResolveProductsResponse, DiskSpaceInfo } from '#lib/api/download.js';
	import { downloadApi } from '#lib/api/download.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { createLogger } from '#lib/logger.js';

	const logger = createLogger('components/metadata/DownloadDialog');

	interface Props {
		open: boolean;
		memberOusUids: string[];
		metadataRows?: MetadataRow[];
		onClose: () => void;
		onStarted: (jobId: string) => void;
	}

	let { open, memberOusUids, metadataRows = [], onClose, onStarted }: Props = $props();

	// State
	let resolving = $state(false);
	let resolved = $state<ResolveProductsResponse | null>(null);
	let resolveError = $state('');

	let productFilter = $state('all');
	let destination = $state('/host_home/Downloads');
	let maxParallel = $state(3);
	let extractTar = $state(false);
	let unpackMs = $state(false);
	let generateCalibratedVisibilities = $state(false);
	let cleanIntermediateFiles = $state(false);
	let archiveOutputRoot = $state('');
	let casaDataRoot = $state('');
	let skipCasaDataUpdate = $state(false);

	let diskSpace = $state<DiskSpaceInfo | null>(null);
	let diskSpaceLoading = $state(false);
	let diskSpaceError = $state('');

	let starting = $state(false);
	let startError = $state('');

	$effect(() => {
		logger.debug({ productFilter }, 'Product filter changed');
	});

	$effect(() => {
		if (generateCalibratedVisibilities) {
			unpackMs = true;
			extractTar = true;
			productFilter = 'all';
		}
		if (unpackMs) {
			extractTar = true;
		}
		if (!generateCalibratedVisibilities) {
			cleanIntermediateFiles = false;
		}
	});

	// Directory browser state
	let browserOpen = $state(false);

	function openBrowser() {
		browserOpen = true;
	}

	function selectDirectory(path: string) {
		logger.debug({ path }, 'Download destination selected');
		destination = path;
		browserOpen = false;
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
			extractTar = false;
			unpackMs = false;
			generateCalibratedVisibilities = false;
			cleanIntermediateFiles = false;
			archiveOutputRoot = '';
			casaDataRoot = '';
			skipCasaDataUpdate = false;
			browserOpen = false;
		}
	});

	async function resolveProducts() {
		logger.info({ count: memberOusUids.length }, 'Resolving ALMA products');
		resolving = true;
		resolveError = '';
		try {
			resolved = await downloadApi.resolveProducts(memberOusUids);
			logger.info(
				{ totalCount: resolved.total_count, totalSize: resolved.total_size_display },
				'Products resolved'
			);
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
		logger.info(
			{
				productFilter,
				destination,
				maxParallel,
				extractTar,
				unpackMs,
				generateCalibratedVisibilities,
				fileCount: filteredInfo.count
			},
			'Starting download'
		);
		starting = true;
		startError = '';
		try {
			const resp = await downloadApi.startDownload({
				memberOusUids,
				productFilter,
				destination,
				maxParallel,
				extractTar,
				unpackMs,
				generateCalibratedVisibilities,
				cleanIntermediateFiles,
				archiveOutputRoot,
				casaDataRoot,
				skipCasaDataUpdate,
				selectedMetadata: metadataRows
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
		<div class="w-full max-w-xl rounded-lg bg-surface shadow-panel">
			<header class="flex items-center justify-between border-b px-6 py-4">
				<h2 class="text-lg font-semibold text-fg">Download ALMA Data</h2>
				<button
					type="button"
					class="text-fg-dim hover:text-gray-800"
					aria-label="Close"
					onclick={onClose}
				>
					✕
				</button>
			</header>

			<div class="space-y-5 px-6 py-6">
				<!-- Observations summary -->
				<p class="text-sm text-fg-muted">
					{memberOusUids.length} observation{memberOusUids.length === 1 ? '' : 's'} selected.
				</p>

				<!-- Loading / Error state -->
				{#if resolving}
					<div class="flex items-center gap-2 text-sm text-fg-dim">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle
								cx="12"
								cy="12"
								r="10"
								stroke="currentColor"
								stroke-width="4"
								class="opacity-25"
							/>
							<path
								fill="currentColor"
								d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
								class="opacity-75"
							/>
						</svg>
						Resolving available products…
					</div>
				{:else if resolveError}
					<div class="rounded-md bg-danger-soft px-4 py-3 text-sm text-danger">
						{resolveError}
						<button type="button" class="ml-2 underline" onclick={resolveProducts}>Retry</button>
					</div>
				{:else if resolved}
					<!-- Product type selector -->
					<div>
						<label class="block text-sm font-medium text-fg-muted mb-2" for="product-filter">
							Product type
						</label>
						<div class="flex flex-wrap gap-2">
							{#each productTypes as pt (pt.value)}
								{@const typeInfo = pt.value === 'all' ? null : resolved.by_type[pt.value]}
								{@const count = pt.value === 'all' ? resolved.total_count : (typeInfo?.count ?? 0)}
								<button
									type="button"
									class="rounded-full px-3 py-1 text-sm border transition-colors"
									class:bg-primary={productFilter === pt.value}
									class:text-white={productFilter === pt.value}
									class:border-blue-600={productFilter === pt.value}
									class:bg-surface={productFilter !== pt.value}
									class:text-fg-muted={productFilter !== pt.value}
									class:border-line={productFilter !== pt.value}
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
					<div class="rounded-md bg-surface p-3 text-sm text-fg-muted">
						<span class="font-medium">{filteredInfo.count}</span> files,
						<span class="font-medium">{filteredInfo.display}</span> total
					</div>

					<!-- Destination path -->
					<div>
						<label class="block text-sm font-medium text-fg-muted mb-1" for="download-dest">
							Download destination
						</label>
						<div class="flex gap-2">
							<input
								type="text"
								id="download-dest"
								bind:value={destination}
								class="flex-1 rounded-md border border-line px-3 py-2 text-sm"
								placeholder="/path/to/download/directory"
							/>
							<button
								type="button"
								class="rounded-md border border-line px-3 py-2 text-sm font-medium text-fg-muted hover:bg-hover whitespace-nowrap"
								onclick={openBrowser}
							>
								Browse…
							</button>
						</div>

						<!-- Inline directory browser -->
					</div>

					<!-- Disk space -->
					{#if diskSpaceLoading}
						<p class="text-xs text-fg-dim">Checking disk space…</p>
					{:else if diskSpaceError}
						<p class="text-xs text-danger">{diskSpaceError}</p>
					{:else if diskSpace}
						<div
							class="rounded-md p-3 text-sm"
							class:bg-ok-soft={diskSpace.sufficient}
							class:text-ok={diskSpace.sufficient}
							class:bg-danger-soft={!diskSpace.sufficient}
							class:text-danger={!diskSpace.sufficient}
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
						<label class="block text-sm font-medium text-fg-muted mb-1" for="max-parallel">
							Parallel downloads
						</label>
						<input
							type="range"
							id="max-parallel"
							min="1"
							max="30"
							bind:value={maxParallel}
							class="w-full"
						/>
						<p class="text-xs text-fg-dim">
							{maxParallel} simultaneous files (load is balanced across the ESO, NRAO and NAOJ ALMA mirrors,
							capped at 10 per mirror)
						</p>
					</div>

					<!-- Extract tar archives -->
					<label class="flex items-center gap-2 cursor-pointer">
						<input
							type="checkbox"
							bind:checked={extractTar}
							disabled={unpackMs || generateCalibratedVisibilities}
							class="rounded border-line"
						/>
						<span class="text-sm text-fg-muted">Extract tar archives after download</span>
					</label>
					<p class="text-xs text-fg-dim -mt-3 ml-6">
						ALMA delivers data as .tar bundles. Enable this to automatically extract FITS files and
						remove the archives.
					</p>

					<div class="space-y-3 rounded-md border border-line p-3">
						<p class="text-sm font-medium text-fg">MeasurementSet products</p>
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={unpackMs}
								disabled={generateCalibratedVisibilities}
								class="rounded border-line"
							/>
							<span class="text-sm text-fg-muted">Create raw MeasurementSets</span>
						</label>
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={generateCalibratedVisibilities}
								class="rounded border-line"
							/>
							<span class="text-sm text-fg-muted">Create calibrated visibilities</span>
						</label>
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={cleanIntermediateFiles}
								disabled={!generateCalibratedVisibilities}
								class="rounded border-line"
							/>
							<span class="text-sm text-fg-muted">Keep only split calibrated MS products</span>
						</label>
						<div class="grid gap-3 sm:grid-cols-2">
							<input
								type="text"
								bind:value={archiveOutputRoot}
								class="rounded-md border border-line px-3 py-2 text-sm"
								placeholder="Archive output root (optional)"
							/>
							<input
								type="text"
								bind:value={casaDataRoot}
								class="rounded-md border border-line px-3 py-2 text-sm"
								placeholder="CASA data root (optional)"
							/>
						</div>
						<label class="flex items-center gap-2 cursor-pointer">
							<input
								type="checkbox"
								bind:checked={skipCasaDataUpdate}
								class="rounded border-line"
							/>
							<span class="text-sm text-fg-muted">Skip CASA runtime data update</span>
						</label>
					</div>

					<!-- Errors -->
					{#if startError}
						<div class="rounded-md bg-danger-soft px-4 py-3 text-sm text-danger">{startError}</div>
					{/if}
				{/if}

				<!-- Actions -->
				<div class="flex items-center justify-end gap-3 pt-2 border-t">
					<button
						type="button"
						class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover"
						onclick={onClose}
					>
						Cancel
					</button>
					<button
						type="button"
						disabled={!resolved ||
							filteredInfo.count === 0 ||
							starting ||
							(diskSpace != null && !diskSpace.sufficient)}
						class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:bg-green-300 disabled:cursor-not-allowed"
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
	<DirectoryPicker
		startPath={destination || '/'}
		title="Choose Download Folder"
		layer="z-[60]"
		onSelect={selectDirectory}
		onClose={() => (browserOpen = false)}
	/>
{/if}
