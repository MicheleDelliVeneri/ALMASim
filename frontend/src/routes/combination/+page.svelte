<script lang="ts">
	import { onMount } from 'svelte';
	import DatacubeFileList from '#lib/components/visualizer/DatacubeFileList.svelte';
	import ImageCanvas from '#lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '#lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '#lib/logger.js';
	import { downloadApi, type BrowseDirectoryResponse } from '#lib/api/download.js';
	import { API_BASE_URL } from '#lib/config.js';

	const logger = createLogger('routes/combination');
	const PRODUCT_PREFIXES = ['int-image-cube_', 'tp-image-cube_', 'tp-int-image-cube_'] as const;
	type ProductFilter = 'all' | 'int' | 'tp' | 'merged';

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
		productKind: Exclude<ProductFilter, 'all'>;
		scale: number;
		panX: number;
		panY: number;
	}

	let loading = $state(false);
	let error = $state<string | null>(null);
	let loadedImages = $state<LoadedImage[]>([]);
	let outputDir = $state('');
	let allFiles = $state<DatacubeFile[]>([]);
	let fileListLoading = $state(false);
	let productFilter = $state<ProductFilter>('all');

	let dirBrowserOpen = $state(false);
	let dirBrowsing = $state(false);
	let dirBrowseResult = $state<BrowseDirectoryResponse | null>(null);
	let dirBrowseError = $state('');

	let linkedView = $state(false);
	let gridLayout = $state<'horizontal' | 'vertical'>('horizontal');

	function getProductKind(name: string): Exclude<ProductFilter, 'all'> | null {
		if (name.startsWith('tp-int-image-cube_')) return 'merged';
		if (name.startsWith('tp-image-cube_')) return 'tp';
		if (name.startsWith('int-image-cube_')) return 'int';
		return null;
	}

	function getProductLabel(kind: Exclude<ProductFilter, 'all'>): string {
		if (kind === 'merged') return 'TP+INT';
		if (kind === 'tp') return 'TP';
		return 'INT';
	}

	function getProductBadgeClass(kind: Exclude<ProductFilter, 'all'>): string {
		if (kind === 'merged') return 'bg-emerald-100 text-emerald-800';
		if (kind === 'tp') return 'bg-amber-100 text-amber-800';
		return 'bg-accent-soft text-accent';
	}

	const combinationFiles = $derived.by(() =>
		allFiles.filter((file) => {
			const kind = getProductKind(file.name);
			if (!kind) return false;
			return productFilter === 'all' ? true : kind === productFilter;
		})
	);

	async function loadFileList(dir?: string) {
		const targetDir = dir || '/host_home';
		logger.debug({ dir: targetDir }, 'Loading combination file list');
		fileListLoading = true;
		outputDir = targetDir;
		if (targetDir === '/host_home') {
			allFiles = [];
			fileListLoading = false;
			return;
		}
		try {
			const url = `${API_BASE_URL}/api/v1/visualizer/files?dir=${encodeURIComponent(targetDir)}`;
			const response = await fetch(url);
			if (!response.ok) throw new Error('Failed to load combination file list');
			const data: FileListResponse = await response.json();
			outputDir = data.output_dir;
			allFiles = data.files.filter((file) =>
				PRODUCT_PREFIXES.some((prefix) => file.name.startsWith(prefix))
			);
			logger.info({ outputDir: data.output_dir, count: allFiles.length }, 'Combination file list loaded');
		} catch (err) {
			logger.error({ err }, 'Failed to load combination file list');
			allFiles = [];
		} finally {
			fileListLoading = false;
		}
	}

	async function browseDir(path: string) {
		dirBrowsing = true;
		dirBrowseError = '';
		try {
			dirBrowseResult = await downloadApi.browseDirectory(path);
		} catch (e) {
			dirBrowseError = e instanceof Error ? e.message : 'Failed to browse directory';
			logger.error({ path, err: e }, 'Failed to browse combination directory');
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
		closeDirBrowser();
		loadFileList(selectedDir);
	}

	async function processFile(filePath: string) {
		const fileName = filePath.split('/').pop() ?? filePath;
		const productKind = getProductKind(fileName);
		if (!productKind) {
			error = 'Unsupported combination product';
			return;
		}

		loading = true;
		error = null;
		try {
			const dirParam = outputDir ? `?dir=${encodeURIComponent(outputDir)}` : '';
			const fileResponse = await fetch(
				`${API_BASE_URL}/api/v1/visualizer/files/${encodeURIComponent(filePath)}${dirParam}`
			);
			if (!fileResponse.ok) {
				throw new Error(`Failed to load file: ${fileResponse.statusText}`);
			}
			const blob = await fileResponse.blob();
			const serverFile = new File([blob], fileName, { type: 'application/octet-stream' });
			const formData = new FormData();
			formData.append('file', serverFile);
			formData.append('method', 'sum');

			const response = await fetch(`${API_BASE_URL}/api/v1/visualizer/integrate`, {
				method: 'POST',
				body: formData
			});
			if (!response.ok) {
				const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
				throw new Error(errorData.detail || `HTTP ${response.status}`);
			}

			const data: ImageData = await response.json();
			loadedImages = [
				...loadedImages,
				{
					...data,
					id: `${fileName}-${Date.now()}`,
					productKind,
					scale: 1.0,
					panX: 0,
					panY: 0
				}
			];
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to process combination product';
			logger.error({ filePath, err }, 'Failed to process combination product');
		} finally {
			loading = false;
		}
	}

	function removeImage(id: string) {
		loadedImages = loadedImages.filter((img) => img.id !== id);
	}

	function clearAll() {
		loadedImages = [];
	}

	function updateScale(id: string, newScale: number) {
		loadedImages = linkedView
			? loadedImages.map((img) => ({ ...img, scale: newScale }))
			: loadedImages.map((img) => (img.id === id ? { ...img, scale: newScale } : img));
	}

	function updatePan(id: string, newPanX: number, newPanY: number) {
		loadedImages = linkedView
			? loadedImages.map((img) => ({ ...img, panX: newPanX, panY: newPanY }))
			: loadedImages.map((img) =>
					img.id === id ? { ...img, panX: newPanX, panY: newPanY } : img
				);
	}

	function handleReset(id: string) {
		loadedImages = linkedView
			? loadedImages.map((img) => ({ ...img, scale: 1.0, panX: 0, panY: 0 }))
			: loadedImages.map((img) =>
					img.id === id ? { ...img, scale: 1.0, panX: 0, panY: 0 } : img
				);
	}

	async function getDefaultOutputDir(): Promise<string | undefined> {
		const requestedDir =
			typeof window !== 'undefined'
				? new URLSearchParams(window.location.search).get('dir')
				: null;
		if (requestedDir) return requestedDir;
		return '/host_home';
	}

	onMount(async () => {
		const initialDir = await getDefaultOutputDir();
		loadFileList(initialDir);
	});
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-6xl space-y-6">
		<div class="rounded-lg bg-surface p-6 shadow-panel">
			<div class="mb-4 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
				<div>
					<h1 class="text-2xl font-bold text-fg">Combination</h1>
					<p class="mt-2 text-fg-muted">
						Load and compare the image-domain combination products: interferometric, total-power, and merged TP+INT cubes.
					</p>
				</div>
				<div class="flex flex-wrap gap-2">
					{#each [
						{ value: 'all', label: 'All' },
						{ value: 'int', label: 'INT' },
						{ value: 'tp', label: 'TP' },
						{ value: 'merged', label: 'TP+INT' }
					] as option}
						<button
							type="button"
							onclick={() => (productFilter = option.value as ProductFilter)}
							class={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
								productFilter === option.value
									? 'bg-sunk text-white'
									: 'bg-surface text-fg-muted hover:bg-gray-200'
							}`}
						>
							{option.label}
						</button>
					{/each}
				</div>
			</div>

			<DatacubeFileList
				files={combinationFiles}
				loading={fileListLoading}
				{outputDir}
				onFileSelect={processFile}
				onRefresh={() => loadFileList(outputDir || undefined)}
				onBrowseRequest={openDirBrowser}
				disabled={loading}
				title="Available Combination Products"
				emptyMessage="No combination `.npz` files found in the selected output directory"
				actionLabel="Load Product"
			/>

			{#if error}
				<div class="mt-4 rounded-md border border-danger-line bg-danger-soft p-3">
					<p class="text-sm text-danger">{error}</p>
				</div>
			{/if}

			{#if loading}
				<div class="py-4 text-center">
					<p class="text-fg-muted">Processing combination product...</p>
				</div>
			{/if}
		</div>

		{#if loadedImages.length > 0}
			<div class="rounded-lg bg-surface p-4 shadow-panel">
				<div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
					<h2 class="text-lg font-semibold text-fg">
						Loaded Combination Products ({loadedImages.length})
					</h2>

					<div class="flex flex-wrap items-center gap-3">
						<label class="flex items-center space-x-2 text-sm">
							<input
								type="checkbox"
								bind:checked={linkedView}
								class="h-4 w-4 rounded border-line text-accent focus:ring-blue-500"
							/>
							<span class="text-fg-muted">Link Pan/Zoom</span>
						</label>

						<div class="flex items-center space-x-2 rounded-md border border-line bg-surface p-1">
							<button
								onclick={() => (gridLayout = 'horizontal')}
								class={`rounded px-3 py-1 text-sm transition-colors ${
									gridLayout === 'horizontal'
										? 'bg-surface text-fg shadow-panel'
										: 'text-fg-muted hover:text-gray-900'
								}`}
								title="Horizontal grid"
							>
								Grid
							</button>
							<button
								onclick={() => (gridLayout = 'vertical')}
								class={`rounded px-3 py-1 text-sm transition-colors ${
									gridLayout === 'vertical'
										? 'bg-surface text-fg shadow-panel'
										: 'text-fg-muted hover:text-gray-900'
								}`}
								title="Vertical stack"
							>
								Stack
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
				class={`grid gap-6 ${
					gridLayout === 'vertical'
						? 'grid-cols-1'
						: loadedImages.length === 1
							? 'grid-cols-1'
							: loadedImages.length === 2
								? 'md:grid-cols-2'
								: 'md:grid-cols-2 lg:grid-cols-3'
				}`}
			>
				{#each loadedImages as image (image.id)}
					<div class="rounded-lg bg-surface p-6 shadow-panel">
						<div class="mb-4 flex items-start justify-between gap-3">
							<div class="min-w-0">
								<div class="mb-2">
									<span class={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${getProductBadgeClass(image.productKind)}`}>
										{getProductLabel(image.productKind)}
									</span>
								</div>
								<h3 class="truncate text-sm font-semibold text-fg-muted" title={image.stats.cube_name}>
									{image.stats.cube_name}
								</h3>
							</div>
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
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4" role="dialog" aria-modal="true">
		<div class="w-full max-w-lg rounded-lg bg-surface shadow-panel">
			<header class="flex items-center justify-between border-b px-5 py-3">
				<h2 class="text-sm font-semibold text-fg">Select Combination Directory</h2>
			</header>

			<div class="space-y-3 px-5 py-4">
				{#if dirBrowsing && !dirBrowseResult}
					<div class="flex items-center justify-center gap-2 py-6 text-sm text-fg-dim">Loading…</div>
				{:else if dirBrowseError}
					<div class="rounded-md bg-danger-soft px-4 py-3 text-sm text-danger">{dirBrowseError}</div>
				{:else if dirBrowseResult}
					<div class="flex items-center gap-2 rounded-md bg-surface px-3 py-2">
						<span class="truncate font-mono text-xs text-fg-muted">{dirBrowseResult.current}</span>
					</div>

					<ul class="max-h-56 overflow-y-auto rounded-md border border-line bg-surface">
						{#if dirBrowseResult.parent}
							<li class="border-b border-line">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-hover" onclick={() => dirBrowseResult?.parent && browseDir(dirBrowseResult.parent)}>
									<span class="text-fg-dim">↩</span>
									<span class="text-fg-dim">..</span>
								</button>
							</li>
						{/if}
						{#each dirBrowseResult.entries as entry}
							<li class="border-b border-line last:border-b-0">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-accent-soft" onclick={() => browseDir(entry.path)}>
									<span class="text-yellow-500">📁</span>
									<span class="truncate text-fg-muted">{entry.name}</span>
								</button>
							</li>
						{/each}
					</ul>
				{/if}
			</div>

			<footer class="flex justify-end gap-3 border-t px-5 py-3">
				<button type="button" class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover" onclick={closeDirBrowser}>Cancel</button>
				<button
					type="button"
					disabled={!dirBrowseResult}
					class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:cursor-not-allowed disabled:bg-blue-300"
					onclick={selectDir}
				>
					Select This Directory
				</button>
			</footer>
		</div>
	</div>
{/if}
