<script lang="ts">
	import { onMount } from 'svelte';
	import DatacubeFileList from '$lib/components/visualizer/DatacubeFileList.svelte';
	import DatacubeUpload from '$lib/components/visualizer/DatacubeUpload.svelte';
	import ImageCanvas from '$lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '$lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '$lib/logger';
	import { downloadApi, type BrowseDirectoryResponse } from '$lib/api/download';

	const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
	const logger = createLogger('routes/visualizer');

	interface ImageData {
		image: number[][];
		stats: {
			shape: number[];
			integrated_shape: number[];
			min: number;
			max: number;
			mean: number;
			std: number;
			cube_name: string;
		};
		method: string;
	}

	interface DatacubeFile {
		name: string;
		path: string;
		size: number;
		modified: number;
	}

	interface FileListResponse {
		files: DatacubeFile[];
		output_dir: string;
	}

	interface LoadedImage extends ImageData {
		id: string;
		scale: number;
		panX: number;
		panY: number;
	}

	let loading = $state(false);
	let error = $state<string | null>(null);
	let loadedImages = $state<LoadedImage[]>([]);
	let integrationMethod = $state<'sum' | 'mean'>('sum');
	let outputDir = $state<string>('');

	// File list
	let fileList = $state<DatacubeFile[]>([]);
	let fileListLoading = $state(false);

	// Directory browser
	let dirBrowserOpen = $state(false);
	let dirBrowsing = $state(false);
	let dirBrowseResult = $state<BrowseDirectoryResponse | null>(null);
	let dirBrowseError = $state('');

	// View controls
	let linkedView = $state(false);
	let gridLayout = $state<'horizontal' | 'vertical'>('horizontal');

	async function loadFileList(dir?: string) {
		const targetDir = dir || '/host_home';
		logger.debug({ dir: targetDir }, 'Loading file list');
		fileListLoading = true;
		outputDir = targetDir;
		if (targetDir === '/host_home') {
			fileList = [];
			fileListLoading = false;
			return;
		}
		let timeoutId: number | undefined;
		try {
			const controller = new AbortController();
			timeoutId = window.setTimeout(() => controller.abort(), 5000);
			const response = await fetch(
				`${API_BASE_URL}/api/v1/visualizer/files?dir=${encodeURIComponent(targetDir)}`,
				{
					signal: controller.signal
				}
			);
			if (!response.ok) throw new Error('Failed to load file list');
			const data: FileListResponse = await response.json();
			outputDir = data.output_dir;
			fileList = data.files;
			logger.info({ outputDir: data.output_dir, count: data.files.length }, 'File list loaded');
		} catch (err) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				error = 'Timed out while loading visualizer files';
			}
			logger.error({ err }, 'Failed to load file list');
			fileList = [];
		} finally {
			if (timeoutId !== undefined) {
				window.clearTimeout(timeoutId);
			}
			fileListLoading = false;
		}
	}

	async function browseDir(path: string) {
		logger.debug({ path }, 'Browsing directory');
		dirBrowsing = true;
		dirBrowseError = '';
		try {
			dirBrowseResult = await downloadApi.browseDirectory(path);
		} catch (e) {
			dirBrowseError = e instanceof Error ? e.message : 'Failed to browse directory';
			logger.error({ path, err: e }, 'Failed to browse directory');
		} finally {
			dirBrowsing = false;
		}
	}

	function openDirBrowser() {
		dirBrowserOpen = true;
		browseDir(outputDir || '/host_home');
	}

	function closeDirBrowser() {
		dirBrowserOpen = false;
		dirBrowseResult = null;
		dirBrowseError = '';
	}

	function selectDir() {
		if (!dirBrowseResult) return;
		const selectedDir = dirBrowseResult.current;
		logger.info({ dir: selectedDir }, 'Directory selected');
		closeDirBrowser();
		loadFileList(selectedDir);
	}

	// Process a file (either uploaded or from server)
	async function processFile(file: File | string) {
		const fileName = typeof file === 'string' ? file.split('/').pop() : file.name;
		logger.info({ fileName, method: integrationMethod }, 'Processing datacube');
		loading = true;
		error = null;
		let fileTimeoutId: number | undefined;
		let integrateTimeoutId: number | undefined;

		try {
			let formData: FormData;

			if (typeof file === 'string') {
				// Load file from server
				const dirParam = outputDir ? `?dir=${encodeURIComponent(outputDir)}` : '';
				const fileController = new AbortController();
				fileTimeoutId = window.setTimeout(() => fileController.abort(), 5000);
				const fileResponse = await fetch(
					`${API_BASE_URL}/api/v1/visualizer/files/${encodeURIComponent(file)}${dirParam}`,
					{
						signal: fileController.signal
					}
				);
				if (!fileResponse.ok) {
					throw new Error(`Failed to load file: ${fileResponse.statusText}`);
				}
				const blob = await fileResponse.blob();
				const serverFile = new File([blob], fileName ?? 'file.npz', {
					type: 'application/octet-stream'
				});
				formData = new FormData();
				formData.append('file', serverFile);
			} else {
				// Use uploaded file
				formData = new FormData();
				formData.append('file', file);
			}

			formData.append('method', integrationMethod);

			const integrateController = new AbortController();
			integrateTimeoutId = window.setTimeout(() => integrateController.abort(), 10000);
			const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/integrate`, {
				method: 'POST',
				body: formData,
				signal: integrateController.signal
			});

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
				throw new Error(errorData.detail || `HTTP ${response.status}`);
			}

			const data: ImageData = await response.json();
			logger.info({ fileName, shape: data.stats.shape }, 'Datacube processed');

			// Add to loaded images array with unique ID
			const newImage: LoadedImage = {
				...data,
				id: `${fileName}-${Date.now()}`,
				scale: 1.0,
				panX: 0,
				panY: 0
			};

			loadedImages = [...loadedImages, newImage];
		} catch (err) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				error = 'Timed out while processing datacube';
			} else {
				error = err instanceof Error ? err.message : 'Failed to process datacube';
			}
			logger.error({ fileName, err }, 'Failed to process datacube');
		} finally {
			if (fileTimeoutId !== undefined) {
				window.clearTimeout(fileTimeoutId);
			}
			if (integrateTimeoutId !== undefined) {
				window.clearTimeout(integrateTimeoutId);
			}
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

		if (!file.name.endsWith('.npz')) {
			logger.warn({ name: file.name }, 'Invalid file type selected');
			error = 'Please select a .npz file';
			return;
		}

		await processFile(file);
	}

	// Handle file selection from list
	async function handleFileSelect(filePath: string) {
		await processFile(filePath);
	}

	// Update scale for a specific image (or all if linked)
	function updateScale(id: string, newScale: number) {
		if (linkedView) {
			// Update all images
			loadedImages = loadedImages.map((img) => ({ ...img, scale: newScale }));
		} else {
			// Update only the specific image
			loadedImages = loadedImages.map((img) => (img.id === id ? { ...img, scale: newScale } : img));
		}
	}

	// Update pan for a specific image (or all if linked)
	function updatePan(id: string, newPanX: number, newPanY: number) {
		if (linkedView) {
			// Update all images
			loadedImages = loadedImages.map((img) => ({ ...img, panX: newPanX, panY: newPanY }));
		} else {
			// Update only the specific image
			loadedImages = loadedImages.map((img) =>
				img.id === id ? { ...img, panX: newPanX, panY: newPanY } : img
			);
		}
	}

	// Reset view for a specific image (or all if linked)
	function handleReset(id: string) {
		if (linkedView) {
			// Reset all images
			loadedImages = loadedImages.map((img) => ({ ...img, scale: 1.0, panX: 0, panY: 0 }));
		} else {
			// Reset only the specific image
			loadedImages = loadedImages.map((img) =>
				img.id === id ? { ...img, scale: 1.0, panX: 0, panY: 0 } : img
			);
		}
	}

	onMount(async () => {
		logger.info('Visualizer page mounted');
		const requestedDir =
			typeof window !== 'undefined'
				? new URLSearchParams(window.location.search).get('dir')
				: null;
		void loadFileList(requestedDir || '/host_home');
	});
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-6xl space-y-6">
		<div class="rounded-lg bg-white p-6 shadow-md">
			<h1 class="mb-4 text-2xl font-bold text-gray-900">Datacube Visualizer</h1>

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
					onMethodChange={(method) => (integrationMethod = method)}
					{loading}
				/>

				{#if error}
					<div class="rounded-md border border-red-200 bg-red-50 p-3">
						<p class="text-sm text-red-800">{error}</p>
					</div>
				{/if}

				{#if loading}
					<div class="py-4 text-center">
						<p class="text-gray-600">Processing datacube...</p>
					</div>
				{/if}
			</div>
		</div>

		{#if loadedImages.length > 0}
			<div class="rounded-lg bg-white p-4 shadow-md">
				<div
					class="flex flex-col space-y-4 md:flex-row md:items-center md:justify-between md:space-y-0"
				>
					<h2 class="text-lg font-semibold text-gray-900">
						Loaded Images ({loadedImages.length})
					</h2>

					<div class="flex flex-wrap items-center gap-3">
						<!-- Linked View Toggle -->
						<label class="flex items-center space-x-2 text-sm">
							<input
								type="checkbox"
								bind:checked={linkedView}
								class="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
							/>
							<span class="text-gray-700">Link Pan/Zoom</span>
						</label>

						<!-- Grid Layout Toggle -->
						<div
							class="flex items-center space-x-2 rounded-md border border-gray-300 bg-gray-50 p-1"
						>
							<button
								onclick={() => (gridLayout = 'horizontal')}
								class="rounded px-3 py-1 text-sm transition-colors {gridLayout === 'horizontal'
									? 'bg-white text-gray-900 shadow-sm'
									: 'text-gray-600 hover:text-gray-900'}"
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
									? 'bg-white text-gray-900 shadow-sm'
									: 'text-gray-600 hover:text-gray-900'}"
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
							class="rounded-md bg-red-100 px-4 py-2 text-sm font-medium text-red-700 transition-colors hover:bg-red-200"
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
					<div class="rounded-lg bg-white p-6 shadow-md">
						<div class="mb-4 flex items-center justify-between">
							<h3
								class="text-sm font-semibold text-gray-700 truncate"
								title={image.stats.cube_name}
							>
								{image.stats.cube_name}
							</h3>
							<button
								onclick={() => removeImage(image.id)}
								class="rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-600 transition-colors hover:bg-red-100 hover:text-red-700"
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
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-lg rounded-lg bg-white shadow-2xl">
			<header class="flex items-center justify-between border-b px-5 py-3">
				<h2 class="text-sm font-semibold text-gray-900">Select Directory</h2>
			</header>

			<div class="space-y-3 px-5 py-4">
				{#if dirBrowsing && !dirBrowseResult}
					<div class="flex items-center justify-center gap-2 py-6 text-sm text-gray-500">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
							<path fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a10 10 0 100 10h-2A8 8 0 014 12z" class="opacity-75" />
						</svg>
						Loading…
					</div>
				{:else if dirBrowseError}
					<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{dirBrowseError}</div>
				{:else if dirBrowseResult}
					<div class="flex items-center gap-2 rounded-md bg-gray-100 px-3 py-2">
						<span class="truncate font-mono text-xs text-gray-600">{dirBrowseResult.current}</span>
						{#if dirBrowsing}
							<svg class="h-3.5 w-3.5 shrink-0 animate-spin text-gray-400" viewBox="0 0 24 24" fill="none">
								<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
								<path fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a10 10 0 100 10h-2A8 8 0 014 12z" class="opacity-75" />
							</svg>
						{/if}
					</div>

					<ul class="max-h-56 overflow-y-auto rounded-md border border-gray-200 bg-white">
						{#if dirBrowseResult.parent}
							<li class="border-b border-gray-100">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-gray-50" onclick={() => browseDir(dirBrowseResult!.parent!)}>
									<span class="text-gray-400">↩</span>
									<span class="text-gray-500">..</span>
								</button>
							</li>
						{/if}
						{#each dirBrowseResult.entries as entry}
							<li class="border-b border-gray-100 last:border-b-0">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-blue-50" onclick={() => browseDir(entry.path)}>
									<span class="text-yellow-500">📁</span>
									<span class="truncate text-gray-700">{entry.name}</span>
								</button>
							</li>
						{/each}
						{#if dirBrowseResult.entries.length === 0 && !dirBrowseResult.parent}
							<li class="px-3 py-4 text-center text-sm text-gray-400">No subdirectories</li>
						{/if}
					</ul>
				{/if}
			</div>

			<footer class="flex justify-end gap-3 border-t px-5 py-3">
				<button type="button" class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50" onclick={closeDirBrowser}>Cancel</button>
				<button
					type="button"
					disabled={!dirBrowseResult}
					class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
					onclick={selectDir}
				>
					Select This Directory
				</button>
			</footer>
		</div>
	</div>
{/if}
