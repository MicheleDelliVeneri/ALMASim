<script lang="ts">
	import { onMount } from 'svelte';
	import DatacubeFileList from '#lib/components/visualizer/DatacubeFileList.svelte';
	import ImageCanvas from '#lib/components/visualizer/ImageCanvas.svelte';
	import ImageStatistics from '#lib/components/visualizer/ImageStatistics.svelte';
	import { createLogger } from '#lib/logger.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { imagingApi, type DeconvolutionResponse, type ImagePreviewPayload } from '#lib/api/imaging.js';
	import { API_BASE_URL } from '#lib/config.js';

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
				badgeClass: 'bg-accent-soft text-accent',
				scale: 1.0,
				panX: 0,
				panY: 0
			},
			{
				...response.restored,
				id: 'restored',
				title: 'Restored Cube',
				description: 'Component model convolved with the clean beam and added back to the residual',
				badgeClass: 'bg-accent-soft text-accent',
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

	function openDirBrowser() {
		dirBrowserOpen = true;
	}

	function selectDir(selectedDir: string) {
		dirBrowserOpen = false;
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
		<div class="rounded-lg bg-surface p-6 shadow-panel">
			<div class="mb-4">
				<h1 class="text-2xl font-bold text-fg">Imaging</h1>
				<p class="mt-2 text-fg-muted">
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

				<div class="space-y-4 rounded-lg border border-line bg-surface p-4">
					<div>
						<h2 class="text-sm font-semibold text-fg">Selected Imaging Inputs</h2>
						<div class="mt-3 space-y-2 text-sm">
							<div>
								<p class="text-fg-dim">Dirty Cube</p>
								<p class="truncate font-mono text-xs text-fg">{selectedDirtyPath || 'Not selected'}</p>
							</div>
							<div>
								<p class="text-fg-dim">Beam Cube</p>
								<p class="truncate font-mono text-xs text-fg">{matchedBeamPath || 'No matching beam cube found'}</p>
							</div>
							<div>
								<p class="text-fg-dim">Reference Clean Cube</p>
								<p class="truncate font-mono text-xs text-fg">{matchedCleanPath || 'No matching clean cube found'}</p>
							</div>
						</div>
					</div>

					<div class="grid gap-4 md:grid-cols-2">
						<label class="space-y-2">
							<span class="text-sm font-medium text-fg-muted">Cycles</span>
							<input
								type="number"
								min="0"
								max="5000"
								step="10"
								bind:value={cycles}
								class="w-full rounded-md border border-line px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							/>
						</label>

						<label class="space-y-2">
							<span class="text-sm font-medium text-fg-muted">Gain</span>
							<input
								type="number"
								min="0.01"
								max="1"
								step="0.01"
								bind:value={gain}
								class="w-full rounded-md border border-line px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							/>
						</label>
					</div>

					<div class="grid gap-4 md:grid-cols-2">
						<label class="space-y-2">
							<span class="text-sm font-medium text-fg-muted">Integration</span>
							<select
								bind:value={integrationMethod}
								class="w-full rounded-md border border-line px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
							>
								<option value="sum">Sum</option>
								<option value="mean">Mean</option>
							</select>
						</label>

						<div class="space-y-2">
							<label class="flex items-center gap-2 text-sm font-medium text-fg-muted">
								<input
									type="checkbox"
									bind:checked={useThreshold}
									class="h-4 w-4 rounded border-line text-accent focus:ring-blue-500"
								/>
								<span>Use Threshold</span>
							</label>
							<input
								type="number"
								min="0"
								step="0.0001"
								bind:value={threshold}
								disabled={!useThreshold}
								class="w-full rounded-md border border-line px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:bg-gray-100"
							/>
						</div>
					</div>

					<div class="flex flex-wrap items-center gap-3">
						<label class="flex items-center gap-2 text-sm text-fg-muted">
							<input
								type="checkbox"
								bind:checked={linkedView}
								class="h-4 w-4 rounded border-line text-accent focus:ring-blue-500"
							/>
							<span>Link Pan/Zoom</span>
						</label>

						<button
							type="button"
							onclick={runImaging}
							disabled={running || !selectedDirtyPath || !matchedBeamPath}
							class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:cursor-not-allowed disabled:bg-blue-300"
						>
							{running ? 'Running…' : resumeStatePath ? 'Run More Cycles' : 'Run Deconvolution'}
						</button>
					</div>

					{#if runMetadata}
						<div class="rounded-md border border-accent-line bg-accent-soft p-3 text-sm text-accent">
							<p>
								Total cycles completed: <span class="font-semibold">{runMetadata.cycles_completed}</span>
								{#if runMetadata.resumed}
									<span class="ml-1 text-accent">(added {runMetadata.cycles_added} more)</span>
								{:else}
									<span class="ml-1 text-accent">(initial run added {runMetadata.cycles_added})</span>
								{/if}
							</p>
							<p class="mt-1">
								Use the component model to compare against the original clean cube, and the restored cube to compare against the convolved clean reference.
							</p>
						</div>
					{/if}

					{#if error}
						<div class="rounded-md border border-danger-line bg-danger-soft p-3 text-sm text-danger">
							{error}
						</div>
					{/if}
				</div>
			</div>
		</div>

		{#if panels.length > 0}
			<div class="grid gap-6 md:grid-cols-2">
				{#each panels as panel (panel.id)}
					<div class="rounded-lg bg-surface p-6 shadow-panel">
						<div class="mb-4 flex items-start justify-between gap-3">
							<div class="min-w-0">
								<span class={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${panel.badgeClass}`}>
									{panel.title}
								</span>
								<p class="mt-2 text-sm text-fg-muted">{panel.description}</p>
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
	<DirectoryPicker
		startPath={outputDir || '/host_home'}
		title="Select Imaging Directory"
		selectLabel="Select This Directory"
		onSelect={selectDir}
		onClose={() => (dirBrowserOpen = false)}
	/>
{/if}
