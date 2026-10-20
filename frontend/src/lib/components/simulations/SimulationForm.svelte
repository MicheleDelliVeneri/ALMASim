<script lang="ts">
	import { BROWSE_ROOT } from '#lib/constants.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { createLogger } from '#lib/logger.js';

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
		msSaveMode: string;
		nLines: number;
		robust: number;
		imagingAlgorithm: 'legacy' | 'ducc0';
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
		onMsSaveModeChange: (value: string) => void;
		onNLinesChange: (value: number) => void;
		onRobustChange: (value: number) => void;
		onImagingAlgorithmChange: (value: 'legacy' | 'ducc0') => void;
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
		msSaveMode,
		nLines,
		robust,
		imagingAlgorithm,
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
		onMsSaveModeChange,
		onNLinesChange,
		onRobustChange,
		onImagingAlgorithmChange,
		onNumSimulationsChange,
		onSourceOffsetXChange,
		onSourceOffsetYChange,
		onBackgroundModeChange,
		onBackgroundLevelChange,
		onBackgroundSeedChange
	}: Props = $props();

	const controlClass =
		'w-full rounded-md border border-slate-600 bg-sunk px-3 py-2 text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500';
	const disabledControlClass =
		'w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-fg-dim placeholder:text-slate-500 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-cyan-500';

	// Directory browser state
	let browserOpen = $state(false);

	function openBrowser() {
		logger.debug({ outputDir }, 'Directory browser opened');
		browserOpen = true;
	}

	function selectOutputDir(path: string) {
		onOutputDirChange(path);
		browserOpen = false;
	}
</script>

<section class="space-y-4 rounded-lg border border-slate-700 bg-slate-950 p-6 shadow-panel">
	<h2 class="text-xl font-semibold text-slate-100">Simulation Configuration</h2>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
		<div>
			<label for="source_type" class="mb-1 block text-sm font-medium text-slate-300">
				Source Type
			</label>
			<select
				id="source_type"
				value={sourceType}
				onchange={(e) => onSourceTypeChange(e.currentTarget.value)}
				class={controlClass}
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
			<label for="n_pix" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{#if nPix !== null}
					Override active: {nPix} × {nPix} pixels
				{:else}
					Leave blank to auto-compute from field of view and angular resolution
				{/if}
			</p>
		</div>

		<div>
			<label for="n_channels" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
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
			<label for="simulation_name" class="mb-1 block text-sm font-medium text-slate-300">
				Simulation Name
			</label>
			<input
				type="text"
				id="simulation_name"
				value={simulationName}
				oninput={(e) => onSimulationNameChange(e.currentTarget.value)}
				placeholder="My Simulation"
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Name for this simulation run</p>
		</div>

		<div>
			<label for="output_dir" class="mb-1 block text-sm font-medium text-slate-300">
				Output Directory
			</label>
			<div class="flex gap-2">
				<input
					type="text"
					id="output_dir"
					value={outputDir}
					oninput={(e) => onOutputDirChange(e.currentTarget.value)}
					placeholder={BROWSE_ROOT}
					class={`flex-1 ${controlClass}`}
				/>
				<button
					type="button"
					onclick={openBrowser}
					class="rounded-md border border-slate-600 bg-sunk px-3 py-2 text-sm font-medium text-fg-muted hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-500"
					title="Browse for directory"
				>
					<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
					</svg>
				</button>
			</div>
			<p class="mt-1 text-xs text-fg-dim">Directory to save results (optional)</p>
		</div>

		<div>
			<label for="num_simulations" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Run multiple simulations</p>
		</div>
	</div>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-5">
		<div>
			<label for="use_metadata_snr" class="mb-1 block text-sm font-medium text-slate-300">
				SNR Source
			</label>
			<select
				id="use_metadata_snr"
				value={useMetadataSnr ? 'metadata' : 'manual'}
				onchange={(e) => onUseMetadataSnrChange(e.currentTarget.value === 'metadata')}
				class={controlClass}
			>
				<option value="metadata">Auto from Metadata</option>
				<option value="manual">Manual Override</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">Default: derive from metadata sensitivities</p>
		</div>

		<div>
			<label for="snr" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={disabledControlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{useMetadataSnr ? 'Disabled while using metadata-derived SNR' : 'Manual SNR override'}
			</p>
		</div>

		<div>
			<label for="use_metadata_pwv" class="mb-1 block text-sm font-medium text-slate-300">
				PWV Source
			</label>
			<select
				id="use_metadata_pwv"
				value={useMetadataPwv ? 'metadata' : 'manual'}
				onchange={(e) => onUseMetadataPwvChange(e.currentTarget.value === 'metadata')}
				class={controlClass}
			>
				<option value="metadata">Use Metadata Row</option>
				<option value="manual">Manual Override</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">Default: metadata row PWV</p>
		</div>

		<div>
			<label for="pwv_override" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={disabledControlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{useMetadataPwv ? 'Disabled while using metadata PWV' : 'Used for all selected rows'}
			</p>
		</div>

		<div>
			<label for="save_mode" class="mb-1 block text-sm font-medium text-slate-300">
				Image Save Mode
			</label>
			<select
				id="save_mode"
				value={saveMode}
				onchange={(e) => onSaveModeChange(e.currentTarget.value)}
				class={controlClass}
			>
				<option value="npz">NPZ (NumPy)</option>
				<option value="fits">FITS</option>
				<option value="both">Both</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">Output format for image cubes</p>
		</div>

		<div>
			<label for="ms_save_mode" class="mb-1 block text-sm font-medium text-slate-300">
				MS Save Mode
			</label>
			<select
				id="ms_save_mode"
				value={msSaveMode}
				onchange={(e) => onMsSaveModeChange(e.currentTarget.value)}
				class={controlClass}
			>
				<option value="npz">NPZ (visibilities)</option>
				<option value="msv2">MSv2 (CASA)</option>
				<option value="both">Both</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">Output format for visibility data</p>
		</div>

		<div>
			<label for="n_lines" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Spectral lines (0 for auto)</p>
		</div>

		<div>
			<label for="robust" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Range: -2 to 2 (default: 0)</p>
		</div>

		<div>
			<label for="imaging_algorithm" class="mb-1 block text-sm font-medium text-slate-300">
				Imaging Algorithm
			</label>
			<select
				id="imaging_algorithm"
				value={imagingAlgorithm}
				onchange={(e) =>
					onImagingAlgorithmChange(e.currentTarget.value as 'legacy' | 'ducc0')}
				class={controlClass}
			>
				<option value="legacy">Legacy</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">Select the interferometric channel imaging backend</p>
		</div>
	</div>

	<div class="grid grid-cols-1 gap-4 md:grid-cols-5">
		<div>
			<label for="source_offset_x_arcsec" class="mb-1 block text-sm font-medium text-slate-300">
				Source Offset X
			</label>
			<input
				type="number"
				id="source_offset_x_arcsec"
				step="0.1"
				value={sourceOffsetXArcsec}
				oninput={(e) => onSourceOffsetXChange(parseFloat(e.currentTarget.value) || 0.0)}
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Arcsec from phase center. Default: 0</p>
		</div>

		<div>
			<label for="source_offset_y_arcsec" class="mb-1 block text-sm font-medium text-slate-300">
				Source Offset Y
			</label>
			<input
				type="number"
				id="source_offset_y_arcsec"
				step="0.1"
				value={sourceOffsetYArcsec}
				oninput={(e) => onSourceOffsetYChange(parseFloat(e.currentTarget.value) || 0.0)}
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Arcsec from phase center. Default: 0</p>
		</div>

		<div>
			<label for="background_mode" class="mb-1 block text-sm font-medium text-slate-300">
				Background Sky
			</label>
			<select
				id="background_mode"
				value={backgroundMode}
				onchange={(e) => onBackgroundModeChange(e.currentTarget.value)}
				class={controlClass}
			>
				<option value="none">None</option>
				<option value="blank_field_dsfg">Faint Dusty Galaxies</option>
				<option value="dusty_diffuse">Diffuse Dust</option>
				<option value="combined">Combined</option>
			</select>
			<p class="mt-1 text-xs text-fg-dim">ALMA-band additive sky background</p>
		</div>

		<div>
			<label for="background_level" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Relative amplitude scaling</p>
		</div>

		<div>
			<label for="background_seed" class="mb-1 block text-sm font-medium text-slate-300">
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
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Optional reproducible background seed</p>
		</div>
	</div>
</section>

{#if browserOpen}
	<DirectoryPicker
		startPath={outputDir || BROWSE_ROOT}
		title="Choose Output Folder"
		allowCreate
		onSelect={selectOutputDir}
		onClose={() => (browserOpen = false)}
	/>
{/if}
