<script lang="ts">
	import { BROWSE_ROOT } from '#lib/constants.js';
	import { onMount } from 'svelte';
	import DatacubeFileList from '#lib/components/visualizer/DatacubeFileList.svelte';
	import DatacubeUpload from '#lib/components/visualizer/DatacubeUpload.svelte';
	import ImageCanvas from '#lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '#lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '#lib/logger.js';
	import { INITIAL_VIEW, applyView } from '#lib/utils/linkedViews.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { ApiTimeoutError } from '#lib/api/client.js';
	import { visualizerApi, type DatacubeFile, type IntegratedImage } from '#lib/api/visualizer.js';

	const logger = createLogger('routes/visualizer');
	const DEFAULT_PROCESS_TIMEOUT_MS = 10_000;
	const FITS_PROCESS_TIMEOUT_MS = 20_000;
	const MSV2_PROCESS_TIMEOUT_MS = 120_000;

	interface LoadedImage extends IntegratedImage {
		id: string;
		scale: number;
		panX: number;
		panY: number;
	}

	let loading = $state(false);
	let error = $state<string | null>(null);
	let loadedImages = $state<LoadedImage[]>([]);
	let integrationMethod = $state<'sum' | 'mean'>('sum');
	let complexComponent = $state<'real' | 'imag' | 'magnitude' | 'phase'>('real');
	let outputDir = $state<string>('');
	let channelStart = $state('');
	let channelEnd = $state('');

	// File list
	let fileList = $state<DatacubeFile[]>([]);
	let fileListLoading = $state(false);

	// Directory browser
	let dirBrowserOpen = $state(false);

	// View controls
	let linkedView = $state(false);
	let gridLayout = $state<'horizontal' | 'vertical'>('horizontal');

	function getProcessingTimeoutMs(file: File | string): number {
		const name = (typeof file === 'string' ? file : file.name).toLowerCase();
		if (name.endsWith('.ms')) return MSV2_PROCESS_TIMEOUT_MS;
		if (name.endsWith('.fits') || name.endsWith('.fit') || name.endsWith('.fts')) {
			return FITS_PROCESS_TIMEOUT_MS;
		}
		return DEFAULT_PROCESS_TIMEOUT_MS;
	}

	async function loadFileList(dir?: string) {
		const targetDir = dir || BROWSE_ROOT;
		logger.debug({ dir: targetDir }, 'Loading file list');
		fileListLoading = true;
		outputDir = targetDir;
		if (targetDir === BROWSE_ROOT) {
			fileList = [];
			fileListLoading = false;
			return;
		}
		try {
			const data = await visualizerApi.listFiles(targetDir, { timeoutMs: 5000 });
			outputDir = data.output_dir;
			fileList = data.files;
			logger.info({ outputDir: data.output_dir, count: data.files.length }, 'File list loaded');
		} catch (err) {
			error =
				err instanceof ApiTimeoutError
					? 'Timed out while loading visualizer files'
					: `Could not list ${targetDir}: ${err instanceof Error ? err.message : 'unknown error'}`;
			logger.error({ err }, 'Failed to load file list');
			fileList = [];
		} finally {
			fileListLoading = false;
		}
	}

	function openDirBrowser() {
		dirBrowserOpen = true;
	}

	function selectDir(selectedDir: string) {
		dirBrowserOpen = false;
		loadFileList(selectedDir);
	}

	// Process a file (either uploaded or from server)
	async function processFile(file: File | string) {
		const fileName = typeof file === 'string' ? file.split('/').pop() : file.name;
		const timeoutMs = getProcessingTimeoutMs(file);
		logger.info(
			{ fileName, method: integrationMethod, timeoutMs },
			'Processing visualizer product'
		);
		loading = true;
		error = null;

		try {
			const data = await visualizerApi.integrate(
				typeof file === 'string' ? { serverPath: file, dir: outputDir || undefined } : { file },
				{
					method: integrationMethod,
					complexComponent,
					integrationAxis: '0',
					channelStart,
					channelEnd
				},
				{ timeoutMs }
			);
			logger.info({ fileName, shape: data.stats.shape }, 'Datacube processed');

			// Add to loaded images array with unique ID
			const newImage: LoadedImage = {
				...data,
				stats: {
					...data.stats,
					cube_name: fileName ?? data.stats.cube_name
				},
				id: `${fileName}-${Date.now()}`,
				scale: 1.0,
				panX: 0,
				panY: 0
			};

			loadedImages = [...loadedImages, newImage];
		} catch (err) {
			if (err instanceof ApiTimeoutError) {
				error =
					timeoutMs === MSV2_PROCESS_TIMEOUT_MS
						? 'Timed out while processing MSv2 file after 2 minutes'
						: 'Timed out while processing file';
			} else {
				error = err instanceof Error ? err.message : 'Failed to process file';
			}
			logger.error({ fileName, err }, 'Failed to process datacube');
		} finally {
			loading = false;
		}
	}

	// Remove an image from the grid
	function removeImage(id: string) {
		logger.debug({ id }, 'Removing image');
		loadedImages = loadedImages.filter((img) => img.id !== id);
	}

	// Clear all images
	function clearAll() {
		logger.debug('Clearing all images');
		loadedImages = [];
	}

	// Handle file upload
	async function handleFileUpload(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		const lowerName = file.name.toLowerCase();
		if (
			!lowerName.endsWith('.npz') &&
			!lowerName.endsWith('.fits') &&
			!lowerName.endsWith('.fit') &&
			!lowerName.endsWith('.fts')
		) {
			logger.warn({ name: file.name }, 'Invalid file type selected');
			error = 'Please select a .npz or FITS file';
			return;
		}

		await processFile(file);
	}

	// Handle file selection from list
	async function handleFileSelect(filePath: string) {
		await processFile(filePath);
	}

	function updateScale(id: string, scale: number) {
		loadedImages = applyView(loadedImages, id, { scale }, linkedView);
	}

	function updatePan(id: string, panX: number, panY: number) {
		loadedImages = applyView(loadedImages, id, { panX, panY }, linkedView);
	}

	function handleReset(id: string) {
		loadedImages = applyView(loadedImages, id, INITIAL_VIEW, linkedView);
	}

	onMount(async () => {
		logger.info('Visualizer page mounted');
		const requestedDir =
			typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('dir') : null;
		void loadFileList(requestedDir || BROWSE_ROOT);
	});
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-6xl space-y-6">
		<div class="rounded-lg bg-surface p-6 shadow-panel">
			<h1 class="mb-4 text-2xl font-bold text-fg">Product Visualizer</h1>
			<p class="mb-4 text-sm text-fg-muted">
				Load ALMASim `.npz`, FITS, or MSv2 (`.ms`) outputs for quick inspection.
			</p>

			<div class="space-y-4">
				<DatacubeFileList
					files={fileList}
					loading={fileListLoading}
					{outputDir}
					onFileSelect={handleFileSelect}
					onRefresh={() => loadFileList(outputDir || undefined)}
					onBrowseRequest={openDirBrowser}
					disabled={loading}
				/>

				<DatacubeUpload
					onFileUpload={handleFileUpload}
					{integrationMethod}
					{complexComponent}
					{channelStart}
					{channelEnd}
					onMethodChange={(method) => (integrationMethod = method)}
					onComplexComponentChange={(component) => (complexComponent = component)}
					onChannelStartChange={(value) => (channelStart = value)}
					onChannelEndChange={(value) => (channelEnd = value)}
					{loading}
				/>

				{#if error}
					<div class="rounded-md border border-danger-line bg-danger-soft p-3">
						<p class="text-sm text-danger">{error}</p>
					</div>
				{/if}

				{#if loading}
					<div class="py-4 text-center">
						<p class="text-fg-muted">Processing file...</p>
						<p class="mt-1 text-sm text-fg-dim">MSv2 products can take up to 2 minutes to open.</p>
					</div>
				{/if}
			</div>
		</div>

		{#if loadedImages.length > 0}
			<div class="rounded-lg bg-surface p-4 shadow-panel">
				<div
					class="flex flex-col space-y-4 md:flex-row md:items-center md:justify-between md:space-y-0"
				>
					<h2 class="text-lg font-semibold text-fg">
						Loaded Images ({loadedImages.length})
					</h2>

					<div class="flex flex-wrap items-center gap-3">
						<!-- Linked View Toggle -->
						<label class="flex items-center space-x-2 text-sm">
							<input
								type="checkbox"
								bind:checked={linkedView}
								class="h-4 w-4 rounded border-line text-accent focus:ring-magenta"
							/>
							<span class="text-fg-muted">Link Pan/Zoom</span>
						</label>

						<!-- Grid Layout Toggle -->
						<div class="flex items-center space-x-2 rounded-md border border-line bg-surface p-1">
							<button
								onclick={() => (gridLayout = 'horizontal')}
								class="rounded px-3 py-1 text-sm transition-colors {gridLayout === 'horizontal'
									? 'bg-surface text-fg shadow-panel'
									: 'text-fg-muted hover:text-gray-900'}"
								title="Horizontal grid"
							>
								<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<rect x="3" y="4" width="8" height="7" rx="1" stroke-width="2"></rect>
									<rect x="13" y="4" width="8" height="7" rx="1" stroke-width="2"></rect>
									<rect x="3" y="13" width="8" height="7" rx="1" stroke-width="2"></rect>
									<rect x="13" y="13" width="8" height="7" rx="1" stroke-width="2"></rect>
								</svg>
							</button>
							<button
								onclick={() => (gridLayout = 'vertical')}
								class="rounded px-3 py-1 text-sm transition-colors {gridLayout === 'vertical'
									? 'bg-surface text-fg shadow-panel'
									: 'text-fg-muted hover:text-gray-900'}"
								title="Vertical stack"
							>
								<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<rect x="4" y="3" width="16" height="7" rx="1" stroke-width="2"></rect>
									<rect x="4" y="14" width="16" height="7" rx="1" stroke-width="2"></rect>
								</svg>
							</button>
						</div>

						<button
							onclick={clearAll}
							class="rounded-md bg-danger-soft px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-red-200"
						>
							Clear All
						</button>
					</div>
				</div>
			</div>

			<div
				class="grid gap-6 {gridLayout === 'vertical'
					? 'grid-cols-1'
					: loadedImages.length === 1
						? 'grid-cols-1'
						: loadedImages.length === 2
							? 'md:grid-cols-2'
							: 'md:grid-cols-2 lg:grid-cols-3'}"
			>
				{#each loadedImages as image (image.id)}
					<div class="rounded-lg bg-surface p-6 shadow-panel">
						<div class="mb-4 flex items-center justify-between">
							<h3
								class="text-sm font-semibold text-fg-muted truncate"
								title={image.stats.cube_name}
							>
								{image.stats.cube_name}
							</h3>
							<button
								onclick={() => removeImage(image.id)}
								class="rounded-md bg-surface px-2 py-1 text-xs text-fg-muted transition-colors hover:bg-danger-soft hover:text-danger"
								title="Remove this image"
							>
								✕
							</button>
						</div>

						<ImageStatistics stats={image.stats} method={image.method} />

						<div class="mt-4">
							<ImageCanvas
								imageData={image}
								scale={image.scale}
								panX={image.panX}
								panY={image.panY}
								onScaleChange={(s) => updateScale(image.id, s)}
								onPanChange={(x, y) => updatePan(image.id, x, y)}
								onReset={() => handleReset(image.id)}
							/>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>

{#if dirBrowserOpen}
	<DirectoryPicker
		startPath={outputDir || BROWSE_ROOT}
		selectLabel="Select This Directory"
		onSelect={selectDir}
		onClose={() => (dirBrowserOpen = false)}
	/>
{/if}
