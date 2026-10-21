<script lang="ts">
	import { BROWSE_ROOT } from '#lib/constants.js';
	import { onMount } from 'svelte';
	import DatacubeFileList from '#lib/components/visualizer/DatacubeFileList.svelte';
	import ImageCanvas from '#lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '#lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '#lib/logger.js';
	import { INITIAL_VIEW, applyView } from '#lib/utils/linkedViews.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { visualizerApi, type DatacubeFile, type IntegratedImage } from '#lib/api/visualizer.js';

	const logger = createLogger('routes/combination');
	const PRODUCT_PREFIXES = ['int-image-cube_', 'tp-image-cube_', 'tp-int-image-cube_'] as const;
	type ProductFilter = 'all' | 'int' | 'tp' | 'merged';

	interface LoadedImage extends IntegratedImage {
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
		const targetDir = dir || BROWSE_ROOT;
		logger.debug({ dir: targetDir }, 'Loading combination file list');
		fileListLoading = true;
		outputDir = targetDir;
		if (targetDir === BROWSE_ROOT) {
			allFiles = [];
			fileListLoading = false;
			return;
		}
		try {
			const data = await visualizerApi.listFiles(targetDir);
			outputDir = data.output_dir;
			allFiles = data.files.filter((file) =>
				PRODUCT_PREFIXES.some((prefix) => file.name.startsWith(prefix))
			);
			logger.info(
				{ outputDir: data.output_dir, count: allFiles.length },
				'Combination file list loaded'
			);
		} catch (err) {
			logger.error({ err }, 'Failed to load combination file list');
			error = `Could not list ${targetDir}: ${err instanceof Error ? err.message : 'unknown error'}`;
			allFiles = [];
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
			const data = await visualizerApi.integrate(
				{ serverPath: filePath, dir: outputDir || undefined },
				{ method: 'sum' }
			);
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

	function updateScale(id: string, scale: number) {
		loadedImages = applyView(loadedImages, id, { scale }, linkedView);
	}

	function updatePan(id: string, panX: number, panY: number) {
		loadedImages = applyView(loadedImages, id, { panX, panY }, linkedView);
	}

	function handleReset(id: string) {
		loadedImages = applyView(loadedImages, id, INITIAL_VIEW, linkedView);
	}

	async function getDefaultOutputDir(): Promise<string | undefined> {
		const requestedDir =
			typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('dir') : null;
		if (requestedDir) return requestedDir;
		return BROWSE_ROOT;
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
						Load and compare the image-domain combination products: interferometric, total-power,
						and merged TP+INT cubes.
					</p>
				</div>
				<div class="flex flex-wrap gap-2">
					{#each [{ value: 'all', label: 'All' }, { value: 'int', label: 'INT' }, { value: 'tp', label: 'TP' }, { value: 'merged', label: 'TP+INT' }] as option (option.value)}
						<button
							type="button"
							onclick={() => (productFilter = option.value as ProductFilter)}
							class={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
								productFilter === option.value
									? 'bg-primary text-white'
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
								class="h-4 w-4 rounded border-line text-accent focus:ring-magenta"
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
									<span
										class={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${getProductBadgeClass(image.productKind)}`}
									>
										{getProductLabel(image.productKind)}
									</span>
								</div>
								<h3
									class="truncate text-sm font-semibold text-fg-muted"
									title={image.stats.cube_name}
								>
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
	<DirectoryPicker
		startPath={outputDir || BROWSE_ROOT}
		title="Select Combination Directory"
		selectLabel="Select This Directory"
		onSelect={selectDir}
		onClose={() => (dirBrowserOpen = false)}
	/>
{/if}
