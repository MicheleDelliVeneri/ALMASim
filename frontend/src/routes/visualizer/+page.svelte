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

	let loading = $state(false);
	let error = $state<string | null>(null);
	let imageData = $state<ImageData | null>(null);
	let integrationMethod = $state<'sum' | 'mean'>('sum');
	let outputDir = $state<string>('');

	// Canvas and zoom/pan state
	let scale = $state(1.0);
	let panX = $state(0);
	let panY = $state(0);

	// File list
	let fileList = $state<DatacubeFile[]>([]);
	let fileListLoading = $state(false);

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

			if (typeof file === 'string') {
				// Load file from server
				const fileResponse = await fetch(
					`${API_BASE_URL}/api/v1/visualizer/files/${encodeURIComponent(file)}`
				);
				if (!fileResponse.ok) {
					throw new Error(`Failed to load file: ${fileResponse.statusText}`);
				}
				const blob = await fileResponse.blob();
				const serverFile = new File([blob], file.split('/').pop() || 'file.npz', {
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

			const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/integrate`, {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
				throw new Error(errorData.detail || `HTTP ${response.status}`);
			}

			const data: ImageData = await response.json();
			imageData = data;
			scale = 1.0;
			panX = 0;
			panY = 0;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to process datacube';
		} finally {
			loading = false;
		}
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

	function handleReset() {
		scale = 1.0;
		panX = 0;
		panY = 0;
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

				{#if imageData}
					<ImageStatistics stats={imageData.stats} method={imageData.method} />
				{/if}
			</div>
		</div>

		{#if imageData}
			<div class="rounded-lg bg-white p-6 shadow-md">
				<ImageCanvas
					{imageData}
					{scale}
					{panX}
					{panY}
					onScaleChange={(s) => (scale = s)}
					onPanChange={(x, y) => {
						panX = x;
						panY = y;
					}}
					onReset={handleReset}
				/>
			</div>
		{/if}
	</div>
</div>
