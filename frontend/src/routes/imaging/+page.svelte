<script lang="ts">
	import { onMount } from 'svelte';
	import DatacubeFileList from '$lib/components/visualizer/DatacubeFileList.svelte';
	import ImageCanvas from '$lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '$lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '$lib/logger';
	import { downloadApi, type BrowseDirectoryResponse } from '$lib/api/download';
	import { imagingApi, type DeconvolutionResponse, type ImagePreviewPayload } from '$lib/api/imaging';
	import { API_BASE_URL } from '$lib/config';

	const logger = createLogger('routes/imaging');
	const DIRTY_PREFIX = 'dirty-cube_';
	const BEAM_PREFIX = 'beam-cube_';
	const CLEAN_PREFIX = 'clean-cube_';

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

	interface PreviewPanel extends ImagePreviewPayload {
		id: string;
		title: string;
		description: string;
		badgeClass: string;
		scale: number;
		panX: number;
		panY: number;
	}

	let loadingFiles = $state(false);
	let running = $state(false);
	let error = $state<string | null>(null);
	let outputDir = $state('');
	let allFiles = $state<DatacubeFile[]>([]);

	let selectedDirtyPath = $state('');
	let matchedBeamPath = $state('');
	let matchedCleanPath = $state('');

	let cycles = $state(150);
	let gain = $state(0.12);
	let integrationMethod = $state<'sum' | 'mean'>('sum');
	let useThreshold = $state(false);
	let threshold = $state(0.0);

	let panels = $state<PreviewPanel[]>([]);
	let runMetadata = $state<DeconvolutionResponse['metadata'] | null>(null);
	let resumeStatePath = $state<string | null>(null);

	let linkedView = $state(true);

	let dirBrowserOpen = $state(false);
	let dirBrowsing = $state(false);
	let dirBrowseResult = $state<BrowseDirectoryResponse | null>(null);
	let dirBrowseError = $state('');

	const dirtyFiles = $derived.by(() =>
		allFiles.filter((file) => file.name.startsWith(DIRTY_PREFIX))
	);

	function normalizeParent(path: string): string {
		const parts = path.split('/');
		return parts.length > 1 ? parts.slice(0, -1).join('/') : '';
	}

	function findCompanion(path: string, prefix: string): string {
		const fileName = path.split('/').pop() ?? path;
		if (!fileName.startsWith(DIRTY_PREFIX)) return '';
		const suffix = fileName.slice(DIRTY_PREFIX.length);
		const parent = normalizeParent(path);
		const expectedPath = parent ? `${parent}/${prefix}${suffix}` : `${prefix}${suffix}`;
		const exactMatch = allFiles.find((file) => file.path === expectedPath);
		if (exactMatch) return exactMatch.path;
		return allFiles.find((file) => file.name === `${prefix}${suffix}`)?.path ?? '';
	}

	function resetPanels() {
		panels = [];
		runMetadata = null;
		resumeStatePath = null;
	}

	function buildPanels(response: DeconvolutionResponse): PreviewPanel[] {
		const built: PreviewPanel[] = [];
		if (response.reference_clean) {
			built.push({
				...response.reference_clean,
				id: 'reference-clean',
				title: 'Reference Clean Cube',
				description: 'Original model/clean cube saved by the simulation',
				badgeClass: 'bg-emerald-100 text-emerald-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			});
		}
		if (response.convolved_reference) {
			built.push({
				...response.convolved_reference,
				id: 'convolved-reference',
				title: 'Convolved Clean Reference',
				description: 'Original clean cube convolved with the clean beam for apples-to-apples comparison with the restored image',
				badgeClass: 'bg-teal-100 text-teal-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			});
		}
		built.push(
			{
				...response.dirty,
				id: 'dirty',
				title: 'Dirty Cube',
				description: 'Observed interferometric cube before deconvolution',
				badgeClass: 'bg-slate-100 text-slate-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			},
			{
				...response.component_model,
				id: 'component-model',
				title: 'Component Model',
				description: 'CLEAN component cube, which should approach the original clean sky model',
				badgeClass: 'bg-indigo-100 text-indigo-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			},
			{
				...response.restored,
				id: 'restored',
				title: 'Restored Cube',
				description: 'Component model convolved with the clean beam and added back to the residual',
				badgeClass: 'bg-blue-100 text-blue-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			},
			{
				...response.residual,
				id: 'residual',
				title: 'Residual Cube',
				description: 'Residual cube after subtracting CLEAN components',
				badgeClass: 'bg-amber-100 text-amber-800',
				scale: 1.0,
				panX: 0,
				panY: 0
			}
		);
		return built;
	}

	function updateScale(id: string, newScale: number) {
		panels = linkedView
			? panels.map((panel) => ({ ...panel, scale: newScale }))
			: panels.map((panel) => (panel.id === id ? { ...panel, scale: newScale } : panel));
	}

	function updatePan(id: string, newPanX: number, newPanY: number) {
		panels = linkedView
			? panels.map((panel) => ({ ...panel, panX: newPanX, panY: newPanY }))
			: panels.map((panel) =>
					panel.id === id ? { ...panel, panX: newPanX, panY: newPanY } : panel
				);
	}

	function handleReset(id: string) {
		panels = linkedView
			? panels.map((panel) => ({ ...panel, scale: 1.0, panX: 0, panY: 0 }))
			: panels.map((panel) =>
					panel.id === id ? { ...panel, scale: 1.0, panX: 0, panY: 0 } : panel
				);
	}

	async function loadFileList(dir?: string) {
		const targetDir = dir || '/host_home';
		loadingFiles = true;
		error = null;
		outputDir = targetDir;
		if (targetDir === '/host_home') {
			allFiles = [];
			selectedDirtyPath = '';
			matchedBeamPath = '';
			matchedCleanPath = '';
			resetPanels();
			loadingFiles = false;
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
			if (!response.ok) {
				throw new Error('Failed to load imaging file list');
			}
			const data: FileListResponse = await response.json();
			outputDir = data.output_dir;
			allFiles = data.files.filter((file) =>
				[DIRTY_PREFIX, BEAM_PREFIX, CLEAN_PREFIX].some((prefix) => file.name.startsWith(prefix))
			);
			if (selectedDirtyPath && !allFiles.some((file) => file.path === selectedDirtyPath)) {
				selectedDirtyPath = '';
				matchedBeamPath = '';
				matchedCleanPath = '';
				resetPanels();
			}
			logger.info({ outputDir: data.output_dir, count: allFiles.length }, 'Imaging file list loaded');
		} catch (err) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				error = 'Timed out while loading imaging files';
			} else {
				error = err instanceof Error ? err.message : 'Failed to load imaging file list';
			}
			allFiles = [];
			logger.error({ err }, 'Failed to load imaging file list');
		} finally {
			if (timeoutId !== undefined) {
				window.clearTimeout(timeoutId);
			}
			loadingFiles = false;
		}
	}

	function selectDirtyFile(path: string) {
		selectedDirtyPath = path;
		matchedBeamPath = findCompanion(path, BEAM_PREFIX);
		matchedCleanPath = findCompanion(path, CLEAN_PREFIX);
		resetPanels();
		error = null;
	}

	async function runImaging() {
		if (!selectedDirtyPath) {
			error = 'Select a dirty cube first';
			return;
		}
		if (!matchedBeamPath) {
			error = 'No matching beam cube was found for the selected dirty cube';
			return;
		}

		running = true;
		error = null;
		try {
			const response = await imagingApi.deconvolve({
				directory: outputDir,
				dirtyCubePath: selectedDirtyPath,
				beamCubePath: matchedBeamPath,
				cleanCubePath: matchedCleanPath || null,
				cycles,
				gain,
				threshold: useThreshold ? threshold : null,
				statePath: resumeStatePath,
				method: integrationMethod
			});
			panels = buildPanels(response);
			runMetadata = response.metadata;
			resumeStatePath = response.metadata.state_path;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to run imaging';
			logger.error(
				{
					err,
					outputDir,
					selectedDirtyPath,
					matchedBeamPath,
					matchedCleanPath,
					cycles,
					gain
				},
				'Failed to run imaging'
			);
		} finally {
			running = false;
		}
	}

	async function browseDir(path: string) {
		dirBrowsing = true;
		dirBrowseError = '';
		try {
			dirBrowseResult = await downloadApi.browseDirectory(path);
		} catch (e) {
			dirBrowseError = e instanceof Error ? e.message : 'Failed to browse directory';
			logger.error({ path, err: e }, 'Failed to browse imaging directory');
		} finally {
			dirBrowsing = false;
		}
	}

	function openDirBrowser() {
		dirBrowserOpen = true;
		void browseDir(outputDir || '/host_home');
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
		void loadFileList(selectedDir);
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
		void loadFileList(initialDir);
	});
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-7xl space-y-6">
		<div class="rounded-lg bg-white p-6 shadow-md">
			<div class="mb-4">
				<h1 class="text-2xl font-bold text-gray-900">Imaging</h1>
				<p class="mt-2 text-gray-600">
					Run the iterative deconvolution stage on saved dirty and beam cubes, inspect the restored cube, and compare it against the simulation clean cube and residuals.
				</p>
			</div>

			<div class="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(20rem,0.9fr)]">
				<div>
					<DatacubeFileList
						files={dirtyFiles}
						loading={loadingFiles}
						{outputDir}
						onFileSelect={selectDirtyFile}
						onRefresh={() => loadFileList(outputDir || undefined)}
						onBrowseRequest={openDirBrowser}
						disabled={running}
						title="Available Dirty Cubes"
						emptyMessage="No `dirty-cube_*.npz` files found. Imaging expects an ALMASim simulation output directory, not raw downloaded ALMA products."
						actionLabel="Select"
					/>
				</div>

				<div class="space-y-4 rounded-lg border border-gray-200 bg-gray-50 p-4">
					<div>
						<h2 class="text-sm font-semibold text-gray-900">Selected Imaging Inputs</h2>
						<div class="mt-3 space-y-2 text-sm">
							<div>
								<p class="text-gray-500">Dirty Cube</p>
								<p class="truncate font-mono text-xs text-gray-900">{selectedDirtyPath || 'Not selected'}</p>
							</div>
							<div>
								<p class="text-gray-500">Beam Cube</p>
								<p class="truncate font-mono text-xs text-gray-900">{matchedBeamPath || 'No matching beam cube found'}</p>
							</div>
							<div>
								<p class="text-gray-500">Reference Clean Cube</p>
								<p class="truncate font-mono text-xs text-gray-900">{matchedCleanPath || 'No matching clean cube found'}</p>
							</div>
						</div>
					</div>

					<div class="grid gap-4 md:grid-cols-2">
						<label class="space-y-2">
							<span class="text-sm font-medium text-gray-700">Cycles</span>
							<input
								type="number"
								min="0"
								max="5000"
								step="10"
								bind:value={cycles}
								class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							/>
						</label>

						<label class="space-y-2">
							<span class="text-sm font-medium text-gray-700">Gain</span>
							<input
								type="number"
								min="0.01"
								max="1"
								step="0.01"
								bind:value={gain}
								class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							/>
						</label>
					</div>

					<div class="grid gap-4 md:grid-cols-2">
						<label class="space-y-2">
							<span class="text-sm font-medium text-gray-700">Integration</span>
							<select
								bind:value={integrationMethod}
								class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							>
								<option value="sum">Sum</option>
								<option value="mean">Mean</option>
							</select>
						</label>

						<div class="space-y-2">
							<label class="flex items-center gap-2 text-sm font-medium text-gray-700">
								<input
									type="checkbox"
									bind:checked={useThreshold}
									class="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
								/>
								<span>Use Threshold</span>
							</label>
							<input
								type="number"
								min="0"
								step="0.0001"
								bind:value={threshold}
								disabled={!useThreshold}
								class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:bg-gray-100"
							/>
						</div>
					</div>

					<div class="flex flex-wrap items-center gap-3">
						<label class="flex items-center gap-2 text-sm text-gray-700">
							<input
								type="checkbox"
								bind:checked={linkedView}
								class="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
							/>
							<span>Link Pan/Zoom</span>
						</label>

						<button
							type="button"
							onclick={runImaging}
							disabled={running || !selectedDirtyPath || !matchedBeamPath}
							class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
						>
							{running ? 'Running…' : resumeStatePath ? 'Run More Cycles' : 'Run Deconvolution'}
						</button>
					</div>

					{#if runMetadata}
						<div class="rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
							<p>
								Total cycles completed: <span class="font-semibold">{runMetadata.cycles_completed}</span>
								{#if runMetadata.resumed}
									<span class="ml-1 text-blue-700">(added {runMetadata.cycles_added} more)</span>
								{:else}
									<span class="ml-1 text-blue-700">(initial run added {runMetadata.cycles_added})</span>
								{/if}
							</p>
							<p class="mt-1">
								Use the component model to compare against the original clean cube, and the restored cube to compare against the convolved clean reference.
							</p>
						</div>
					{/if}

					{#if error}
						<div class="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
							{error}
						</div>
					{/if}
				</div>
			</div>
		</div>

		{#if panels.length > 0}
			<div class="grid gap-6 md:grid-cols-2">
				{#each panels as panel (panel.id)}
					<div class="rounded-lg bg-white p-6 shadow-md">
						<div class="mb-4 flex items-start justify-between gap-3">
							<div class="min-w-0">
								<span class={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${panel.badgeClass}`}>
									{panel.title}
								</span>
								<p class="mt-2 text-sm text-gray-600">{panel.description}</p>
							</div>
						</div>

						<ImageStatistics stats={panel.stats} method={panel.method} />

						<div class="mt-4">
							<ImageCanvas
								imageData={panel}
								scale={panel.scale}
								panX={panel.panX}
								panY={panel.panY}
								onScaleChange={(scale) => updateScale(panel.id, scale)}
								onPanChange={(x, y) => updatePan(panel.id, x, y)}
								onReset={() => handleReset(panel.id)}
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
		<div class="w-full max-w-lg rounded-lg bg-white shadow-2xl">
			<header class="flex items-center justify-between border-b px-5 py-3">
				<h2 class="text-sm font-semibold text-gray-900">Select Imaging Directory</h2>
			</header>

			<div class="space-y-3 px-5 py-4">
				{#if dirBrowsing && !dirBrowseResult}
					<div class="flex items-center justify-center gap-2 py-6 text-sm text-gray-500">Loading…</div>
				{:else if dirBrowseError}
					<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{dirBrowseError}</div>
				{:else if dirBrowseResult}
					<div class="flex items-center gap-2 rounded-md bg-gray-100 px-3 py-2">
						<span class="truncate font-mono text-xs text-gray-600">{dirBrowseResult.current}</span>
					</div>

					<ul class="max-h-56 overflow-y-auto rounded-md border border-gray-200 bg-white">
						{#if dirBrowseResult.parent}
							<li class="border-b border-gray-100">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-gray-50" onclick={() => dirBrowseResult?.parent && browseDir(dirBrowseResult.parent)}>
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
