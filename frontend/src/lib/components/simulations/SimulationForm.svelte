<script lang="ts">
	import { downloadApi, type BrowseDirectoryResponse } from '$lib/api/download';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('components/simulations/SimulationForm');

	interface Props {
		sourceType: string;
		nPix: number | null;
		nChannels: number | null;
		simulationName: string;
		outputDir: string;
		snr: number;
		useMetadataSnr: boolean;
		useMetadataPwv: boolean;
		pwvOverride: number;
		saveMode: string;
		nLines: number;
		robust: number;
		numSimulations: number;
		sourceOffsetXArcsec: number;
		sourceOffsetYArcsec: number;
		backgroundMode: string;
		backgroundLevel: number;
		backgroundSeed: number | null;
		onSourceTypeChange: (type: string) => void;
		onNPixChange: (value: number | null) => void;
		onNChannelsChange: (value: number | null) => void;
		onSimulationNameChange: (value: string) => void;
		onOutputDirChange: (value: string) => void;
		onSnrChange: (value: number) => void;
		onUseMetadataSnrChange: (value: boolean) => void;
		onUseMetadataPwvChange: (value: boolean) => void;
		onPwvOverrideChange: (value: number) => void;
		onSaveModeChange: (value: string) => void;
		onNLinesChange: (value: number) => void;
		onRobustChange: (value: number) => void;
		onNumSimulationsChange: (value: number) => void;
		onSourceOffsetXChange: (value: number) => void;
		onSourceOffsetYChange: (value: number) => void;
		onBackgroundModeChange: (value: string) => void;
		onBackgroundLevelChange: (value: number) => void;
		onBackgroundSeedChange: (value: number | null) => void;
	}

	let {
		sourceType,
		nPix,
		nChannels,
		simulationName,
		outputDir,
		snr,
		useMetadataSnr,
		useMetadataPwv,
		pwvOverride,
		saveMode,
		nLines,
		robust,
		numSimulations,
		sourceOffsetXArcsec,
		sourceOffsetYArcsec,
		backgroundMode,
		backgroundLevel,
		backgroundSeed,
		onSourceTypeChange,
		onNPixChange,
		onNChannelsChange,
		onSimulationNameChange,
		onOutputDirChange,
		onSnrChange,
		onUseMetadataSnrChange,
		onUseMetadataPwvChange,
		onPwvOverrideChange,
		onSaveModeChange,
		onNLinesChange,
		onRobustChange,
		onNumSimulationsChange,
		onSourceOffsetXChange,
		onSourceOffsetYChange,
		onBackgroundModeChange,
		onBackgroundLevelChange,
		onBackgroundSeedChange
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
				value={nPix ?? ''}
				placeholder="Auto from metadata"
				oninput={(e) => {
					const value = e.currentTarget.value.trim();
					if (value === '') {
						onNPixChange(null);
						return;
					}
					const parsed = parseInt(value, 10);
					onNPixChange(Number.isNaN(parsed) ? null : parsed);
				}}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">
				{#if nPix !== null}
					Override active: {nPix} × {nPix} pixels
				{:else}
					Leave blank to auto-compute from field of view and angular resolution
				{/if}
			</p>
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
				value={nChannels ?? ''}
				placeholder="Auto from metadata"
				oninput={(e) => {
					const value = e.currentTarget.value.trim();
					if (value === '') {
						onNChannelsChange(null);
						return;
					}
					const parsed = parseInt(value, 10);
					onNChannelsChange(Number.isNaN(parsed) ? null : parsed);
				}}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">
				{#if nChannels !== null}
					Override active: {nChannels} channels
				{:else}
					Leave blank to auto-compute from the metadata frequency support
				{/if}
			</p>
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

	<div class="grid grid-cols-1 gap-4 md:grid-cols-5">
		<div>
			<label for="use_metadata_snr" class="mb-1 block text-sm font-medium text-gray-700">
				SNR Source
			</label>
			<select
				id="use_metadata_snr"
				value={useMetadataSnr ? 'metadata' : 'manual'}
				onchange={(e) => onUseMetadataSnrChange(e.currentTarget.value === 'metadata')}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="metadata">Auto from Metadata</option>
				<option value="manual">Manual Override</option>
			</select>
			<p class="mt-1 text-xs text-gray-500">Default: derive from metadata sensitivities</p>
		</div>

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
				disabled={useMetadataSnr}
				oninput={(e) => onSnrChange(parseFloat(e.currentTarget.value) || 1.3)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-500"
			/>
			<p class="mt-1 text-xs text-gray-500">
				{useMetadataSnr ? 'Disabled while using metadata-derived SNR' : 'Manual SNR override'}
			</p>
		</div>

		<div>
			<label for="use_metadata_pwv" class="mb-1 block text-sm font-medium text-gray-700">
				PWV Source
			</label>
			<select
				id="use_metadata_pwv"
				value={useMetadataPwv ? 'metadata' : 'manual'}
				onchange={(e) => onUseMetadataPwvChange(e.currentTarget.value === 'metadata')}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="metadata">Use Metadata Row</option>
				<option value="manual">Manual Override</option>
			</select>
			<p class="mt-1 text-xs text-gray-500">Default: metadata row PWV</p>
		</div>

		<div>
			<label for="pwv_override" class="mb-1 block text-sm font-medium text-gray-700">
				PWV Override (mm)
			</label>
			<input
				type="number"
				id="pwv_override"
				min="0"
				max="20"
				step="0.1"
				value={pwvOverride}
				disabled={useMetadataPwv}
				oninput={(e) => onPwvOverrideChange(parseFloat(e.currentTarget.value) || 1.0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-500"
			/>
			<p class="mt-1 text-xs text-gray-500">
				{useMetadataPwv ? 'Disabled while using metadata PWV' : 'Used for all selected rows'}
			</p>
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

	<div class="grid grid-cols-1 gap-4 md:grid-cols-5">
		<div>
			<label for="source_offset_x_arcsec" class="mb-1 block text-sm font-medium text-gray-700">
				Source Offset X
			</label>
			<input
				type="number"
				id="source_offset_x_arcsec"
				step="0.1"
				value={sourceOffsetXArcsec}
				oninput={(e) => onSourceOffsetXChange(parseFloat(e.currentTarget.value) || 0.0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Arcsec from phase center. Default: 0</p>
		</div>

		<div>
			<label for="source_offset_y_arcsec" class="mb-1 block text-sm font-medium text-gray-700">
				Source Offset Y
			</label>
			<input
				type="number"
				id="source_offset_y_arcsec"
				step="0.1"
				value={sourceOffsetYArcsec}
				oninput={(e) => onSourceOffsetYChange(parseFloat(e.currentTarget.value) || 0.0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Arcsec from phase center. Default: 0</p>
		</div>

		<div>
			<label for="background_mode" class="mb-1 block text-sm font-medium text-gray-700">
				Background Sky
			</label>
			<select
				id="background_mode"
				value={backgroundMode}
				onchange={(e) => onBackgroundModeChange(e.currentTarget.value)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="none">None</option>
				<option value="blank_field_dsfg">Faint Dusty Galaxies</option>
				<option value="dusty_diffuse">Diffuse Dust</option>
				<option value="combined">Combined</option>
			</select>
			<p class="mt-1 text-xs text-gray-500">ALMA-band additive sky background</p>
		</div>

		<div>
			<label for="background_level" class="mb-1 block text-sm font-medium text-gray-700">
				Background Level
			</label>
			<input
				type="number"
				id="background_level"
				min="0"
				max="10"
				step="0.1"
				value={backgroundLevel}
				oninput={(e) => onBackgroundLevelChange(parseFloat(e.currentTarget.value) || 0.0)}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Relative amplitude scaling</p>
		</div>

		<div>
			<label for="background_seed" class="mb-1 block text-sm font-medium text-gray-700">
				Background Seed
			</label>
			<input
				type="number"
				id="background_seed"
				step="1"
				value={backgroundSeed ?? ''}
				placeholder="Random"
				oninput={(e) => {
					const value = e.currentTarget.value.trim();
					if (value === '') {
						onBackgroundSeedChange(null);
						return;
					}
					const parsed = parseInt(value, 10);
					onBackgroundSeedChange(Number.isNaN(parsed) ? null : parsed);
				}}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			/>
			<p class="mt-1 text-xs text-gray-500">Optional reproducible background seed</p>
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
