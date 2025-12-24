<script lang="ts">
	import { onMount } from 'svelte';
	import DatacubeFileList from '$lib/components/visualizer/DatacubeFileList.svelte';
	import DatacubeUpload from '$lib/components/visualizer/DatacubeUpload.svelte';
	import ImageCanvas from '$lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '$lib/components/visualizer/ImageStatistics.svelte';

	const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

	// View controls
	let linkedView = $state(false);
	let gridLayout = $state<'horizontal' | 'vertical'>('horizontal');

	async function loadFileList() {
		fileListLoading = true;
		try {
			const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/files`);
			if (!response.ok) throw new Error('Failed to load file list');
			const data: FileListResponse = await response.json();
			outputDir = data.output_dir;
			fileList = data.files;
		} catch (err) {
			console.error('Failed to load file list:', err);
			fileList = [];
		} finally {
			fileListLoading = false;
		}
	}

	// Process a file (either uploaded or from server)
	async function processFile(file: File | string) {
		loading = true;
		error = null;

		try {
			let formData: FormData;
			let fileName: string;

			if (typeof file === 'string') {
				// Load file from server
				fileName = file.split('/').pop() || 'file.npz';
				const fileResponse = await fetch(
					`${API_BASE_URL}/api/v1/visualizer/files/${encodeURIComponent(file)}`
				);
				if (!fileResponse.ok) {
					throw new Error(`Failed to load file: ${fileResponse.statusText}`);
				}
				const blob = await fileResponse.blob();
				const serverFile = new File([blob], fileName, {
					type: 'application/octet-stream'
				});
				formData = new FormData();
				formData.append('file', serverFile);
			} else {
				// Use uploaded file
				fileName = file.name;
				formData = new FormData();
				formData.append('file', file);
			}

			formData.append('method', integrationMethod);

			const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/integrate`, {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
				throw new Error(errorData.detail || `HTTP ${response.status}`);
			}

			const data: ImageData = await response.json();

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
			error = err instanceof Error ? err.message : 'Failed to process datacube';
		} finally {
			loading = false;
		}
	}

	// Remove an image from the grid
	function removeImage(id: string) {
		loadedImages = loadedImages.filter((img) => img.id !== id);
	}

	// Clear all images
	function clearAll() {
		loadedImages = [];
	}

	// Handle file upload
	async function handleFileUpload(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		if (!file.name.endsWith('.npz')) {
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

	onMount(() => {
		loadFileList();
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
					onRefresh={loadFileList}
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
