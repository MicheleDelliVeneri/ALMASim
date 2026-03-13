<script lang="ts">
	import { downloadApi, type BrowseDirectoryResponse } from '$lib/api/download';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('components/simulations/SimulationForm');

	interface Props {
		sourceType: string;
		nPix: number;
		nChannels: number;
		simulationName: string;
		outputDir: string;
		snr: number;
		saveMode: string;
		nLines: number;
		robust: number;
		numSimulations: number;
		onSourceTypeChange: (type: string) => void;
		onNPixChange: (value: number) => void;
		onNChannelsChange: (value: number) => void;
		onSimulationNameChange: (value: string) => void;
		onOutputDirChange: (value: string) => void;
		onSnrChange: (value: number) => void;
		onSaveModeChange: (value: string) => void;
		onNLinesChange: (value: number) => void;
		onRobustChange: (value: number) => void;
		onNumSimulationsChange: (value: number) => void;
	}

	let {
		sourceType,
		nPix,
		nChannels,
		simulationName,
		outputDir,
		snr,
		saveMode,
		nLines,
		robust,
		numSimulations,
		onSourceTypeChange,
		onNPixChange,
		onNChannelsChange,
		onSimulationNameChange,
		onOutputDirChange,
		onSnrChange,
		onSaveModeChange,
		onNLinesChange,
		onRobustChange,
		onNumSimulationsChange
	}: Props = $props();

	// Directory browser state
	let browserOpen = $state(false);
	let browsing = $state(false);
	let browseResult = $state<BrowseDirectoryResponse | null>(null);
	let browseError = $state('');
	let newFolderName = $state('');
	let creatingFolder = $state(false);

	async function browseDir(path: string) {
		logger.debug({ path }, 'Browsing directory');
		browsing = true;
		browseError = '';
		try {
			browseResult = await downloadApi.browseDirectory(path);
		} catch (e) {
			browseError = e instanceof Error ? e.message : 'Failed to browse directory';
			logger.error({ err: e, path }, 'Failed to browse directory');
		} finally {
			browsing = false;
		}
	}

	function openBrowser() {
		logger.debug({ outputDir }, 'Directory browser opened');
		browserOpen = true;
		newFolderName = '';
		browseDir(outputDir || '/host_home');
	}

	function closeBrowser() {
		browserOpen = false;
		browseResult = null;
		browseError = '';
		newFolderName = '';
	}

	async function createFolder() {
		if (!browseResult || !newFolderName.trim()) return;
		creatingFolder = true;
		const newPath = `${browseResult.current}/${newFolderName.trim()}`;
		logger.info({ newPath }, 'Creating new folder');
		try {
			browseResult = await downloadApi.createDirectory(newPath);
			newFolderName = '';
			logger.info({ newPath }, 'Folder created');
		} catch (e) {
			browseError = e instanceof Error ? e.message : 'Failed to create folder';
			logger.error({ err: e, newPath }, 'Failed to create folder');
		} finally {
			creatingFolder = false;
		}
	}

	function selectCurrent() {
		if (!browseResult) return;
		logger.info({ path: browseResult.current }, 'Output directory selected');
		onOutputDirChange(browseResult.current);
		closeBrowser();
	}
</script>

<section class="space-y-4 rounded-lg bg-white p-6 shadow-md">
	<h2 class="text-xl font-semibold text-gray-900">Simulation Configuration</h2>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<label for="source_type" class="mb-1 block text-sm font-medium text-gray-700">
				Source Type
			</label>
			<select
				id="source_type"
				value={sourceType}
				onchange={(e) => onSourceTypeChange(e.currentTarget.value)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="point">Point</option>
				<option value="gaussian">Gaussian</option>
				<option value="extended">Extended</option>
				<option value="diffuse">Diffuse</option>
				<option value="galaxy-zoo">Galaxy Zoo</option>
				<option value="molecular">Molecular</option>
				<option value="hubble-100">Hubble</option>
			</select>
		</div>

		<div>
			<label for="n_pix" class="mb-1 block text-sm font-medium text-gray-700">
				Spatial Pixels (n_pix × n_pix)
			</label>
			<input
				type="number"
				id="n_pix"
				min="32"
				max="2048"
				step="32"
				value={nPix}
				oninput={(e) => onNPixChange(parseInt(e.currentTarget.value) || 256)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Cube will be {nPix} × {nPix} × {nChannels}</p>
		</div>

		<div>
			<label for="n_channels" class="mb-1 block text-sm font-medium text-gray-700">
				Spectral Channels
			</label>
			<input
				type="number"
				id="n_channels"
				min="16"
				max="1024"
				step="16"
				value={nChannels}
				oninput={(e) => onNChannelsChange(parseInt(e.currentTarget.value) || 128)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Total channels: {nChannels}</p>
		</div>
	</div>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<label for="simulation_name" class="mb-1 block text-sm font-medium text-gray-700">
				Simulation Name
			</label>
			<input
				type="text"
				id="simulation_name"
				value={simulationName}
				oninput={(e) => onSimulationNameChange(e.currentTarget.value)}
				placeholder="My Simulation"
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Name for this simulation run</p>
		</div>

		<div>
			<label for="output_dir" class="mb-1 block text-sm font-medium text-gray-700">
				Output Directory
			</label>
			<div class="flex gap-2">
				<input
					type="text"
					id="output_dir"
					value={outputDir}
					oninput={(e) => onOutputDirChange(e.currentTarget.value)}
					placeholder="/host_home"
					class="flex-1 rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
				/>
				<button
					type="button"
					onclick={openBrowser}
					class="rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
					title="Browse for directory"
				>
					<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
					</svg>
				</button>
			</div>
			<p class="mt-1 text-xs text-gray-500">Directory to save results (optional)</p>
		</div>

		<div>
			<label for="num_simulations" class="mb-1 block text-sm font-medium text-gray-700">
				Number of Simulations
			</label>
			<input
				type="number"
				id="num_simulations"
				min="1"
				max="100"
				step="1"
				value={numSimulations}
				oninput={(e) => onNumSimulationsChange(parseInt(e.currentTarget.value) || 1)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Run multiple simulations</p>
		</div>
	</div>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-4">
		<div>
			<label for="snr" class="mb-1 block text-sm font-medium text-gray-700">
				Signal-to-Noise Ratio (SNR)
			</label>
			<input
				type="number"
				id="snr"
				min="0.1"
				max="100"
				step="0.1"
				value={snr}
				oninput={(e) => onSnrChange(parseFloat(e.currentTarget.value) || 1.3)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Default: 1.3</p>
		</div>

		<div>
			<label for="save_mode" class="mb-1 block text-sm font-medium text-gray-700">
				Save Mode
			</label>
			<select
				id="save_mode"
				value={saveMode}
				onchange={(e) => onSaveModeChange(e.currentTarget.value)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="npz">NPZ (NumPy)</option>
				<option value="fits">FITS</option>
				<option value="both">Both</option>
			</select>
			<p class="mt-1 text-xs text-gray-500">Output format</p>
		</div>

		<div>
			<label for="n_lines" class="mb-1 block text-sm font-medium text-gray-700">
				Number of Lines
			</label>
			<input
				type="number"
				id="n_lines"
				min="0"
				max="50"
				step="1"
				value={nLines}
				oninput={(e) => onNLinesChange(parseInt(e.currentTarget.value) || 0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Spectral lines (0 for auto)</p>
		</div>

		<div>
			<label for="robust" class="mb-1 block text-sm font-medium text-gray-700">
				Robust Weighting
			</label>
			<input
				type="number"
				id="robust"
				min="-2"
				max="2"
				step="0.1"
				value={robust}
				oninput={(e) => onRobustChange(parseFloat(e.currentTarget.value) || 0.0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Range: -2 to 2 (default: 0)</p>
		</div>
	</div>
</section>

{#if browserOpen}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-lg rounded-lg bg-white shadow-2xl">
			<header class="flex items-center justify-between border-b px-5 py-3">
				<h3 class="text-sm font-semibold text-gray-900">Choose Output Folder</h3>
				<button type="button" class="text-gray-400 hover:text-gray-700" aria-label="Close" onclick={closeBrowser}>✕</button>
			</header>

			<div class="px-5 py-4 space-y-3">
				{#if browsing && !browseResult}
					<div class="flex items-center justify-center gap-2 py-6 text-sm text-gray-500">
						<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
							<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
						</svg>
						Loading…
					</div>
				{:else if browseError}
					<div class="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{browseError}</div>
				{:else if browseResult}
					<!-- Current path -->
					<div class="flex items-center gap-2 rounded-md bg-gray-100 px-3 py-2">
						<span class="truncate font-mono text-xs text-gray-600">{browseResult.current}</span>
						{#if browsing}
							<svg class="h-3.5 w-3.5 shrink-0 animate-spin text-gray-400" viewBox="0 0 24 24" fill="none">
								<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
								<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
							</svg>
						{/if}
					</div>

					<!-- Directory listing -->
					<ul class="max-h-56 overflow-y-auto rounded-md border border-gray-200 bg-white">
						{#if browseResult.parent}
							<li class="border-b border-gray-100">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-gray-50" onclick={() => browseDir(browseResult!.parent!)}>
									<span class="text-gray-400">↩</span>
									<span class="text-gray-500">..</span>
								</button>
							</li>
						{/if}
						{#each browseResult.entries as entry}
							<li class="border-b border-gray-100 last:border-b-0">
								<button type="button" class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm hover:bg-blue-50" onclick={() => browseDir(entry.path)}>
									<span class="text-yellow-500">📁</span>
									<span class="truncate text-gray-700">{entry.name}</span>
								</button>
							</li>
						{:else}
							<li class="px-3 py-4 text-center text-sm italic text-gray-400">No subdirectories</li>
						{/each}
					</ul>

					<!-- New folder row -->
					<div class="flex gap-2">
						<input
							type="text"
							bind:value={newFolderName}
							placeholder="New folder name…"
							class="flex-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							onkeydown={(e) => e.key === 'Enter' && createFolder()}
						/>
						<button
							type="button"
							disabled={!newFolderName.trim() || creatingFolder}
							onclick={createFolder}
							class="rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-50"
						>
							{creatingFolder ? '…' : '+ Create'}
						</button>
					</div>
				{/if}
			</div>

			<footer class="flex items-center justify-end gap-3 border-t px-5 py-3">
				<button type="button" class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50" onclick={closeBrowser}>Cancel</button>
				<button
					type="button"
					disabled={!browseResult}
					class="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
					onclick={selectCurrent}
				>
					Select this folder
				</button>
			</footer>
		</div>
	</div>
{/if}
