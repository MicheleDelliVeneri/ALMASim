<script lang="ts">
	import { BROWSE_ROOT } from '#lib/constants.js';
	import DirectoryPicker from '#lib/components/shared/DirectoryPicker.svelte';
	import { createLogger } from '#lib/logger.js';
	import { parseOptionalInt, type SimulationFormValues } from '#lib/utils/simulationForm.js';

	const logger = createLogger('components/simulations/SimulationForm');

	interface Props {
		/** The form's values, bound by the page (MADIV-28 T7.5). */
		values: SimulationFormValues;
	}

	let { values = $bindable() }: Props = $props();

	const controlClass =
		'w-full rounded-md border border-slate-600 bg-sunk px-3 py-2 text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500';
	const disabledControlClass =
		'w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-fg-dim placeholder:text-slate-500 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-cyan-500';

	// Directory browser state
	let browserOpen = $state(false);

	function openBrowser() {
		logger.debug({ outputDir: values.outputDir }, 'Directory browser opened');
		browserOpen = true;
	}

	function selectOutputDir(path: string) {
		values.outputDir = path;
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
				value={values.sourceType}
				onchange={(e) => (values.sourceType = e.currentTarget.value)}
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
				value={values.nPix ?? ''}
				placeholder="Auto from metadata"
				oninput={(e) => (values.nPix = parseOptionalInt(e.currentTarget.value))}
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{#if values.nPix !== null}
					Override active: {values.nPix} × {values.nPix} pixels
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
				value={values.nChannels ?? ''}
				placeholder="Auto from metadata"
				oninput={(e) => (values.nChannels = parseOptionalInt(e.currentTarget.value))}
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{#if values.nChannels !== null}
					Override active: {values.nChannels} channels
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
				value={values.simulationName}
				oninput={(e) => (values.simulationName = e.currentTarget.value)}
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
					value={values.outputDir}
					oninput={(e) => (values.outputDir = e.currentTarget.value)}
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
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
						/>
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
				value={values.numSimulations}
				oninput={(e) => (values.numSimulations = parseInt(e.currentTarget.value) || 1)}
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
				value={values.useMetadataSnr ? 'metadata' : 'manual'}
				onchange={(e) => (values.useMetadataSnr = e.currentTarget.value === 'metadata')}
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
				value={values.snr}
				disabled={values.useMetadataSnr}
				oninput={(e) => (values.snr = parseFloat(e.currentTarget.value) || 1.3)}
				class={disabledControlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{values.useMetadataSnr
					? 'Disabled while using metadata-derived SNR'
					: 'Manual SNR override'}
			</p>
		</div>

		<div>
			<label for="use_metadata_pwv" class="mb-1 block text-sm font-medium text-slate-300">
				PWV Source
			</label>
			<select
				id="use_metadata_pwv"
				value={values.useMetadataPwv ? 'metadata' : 'manual'}
				onchange={(e) => (values.useMetadataPwv = e.currentTarget.value === 'metadata')}
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
				value={values.pwvOverride}
				disabled={values.useMetadataPwv}
				oninput={(e) => (values.pwvOverride = parseFloat(e.currentTarget.value) || 1.0)}
				class={disabledControlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">
				{values.useMetadataPwv ? 'Disabled while using metadata PWV' : 'Used for all selected rows'}
			</p>
		</div>

		<div>
			<label for="save_mode" class="mb-1 block text-sm font-medium text-slate-300">
				Image Save Mode
			</label>
			<select
				id="save_mode"
				value={values.saveMode}
				onchange={(e) => (values.saveMode = e.currentTarget.value)}
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
				value={values.msSaveMode}
				onchange={(e) => (values.msSaveMode = e.currentTarget.value)}
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
				value={values.nLines}
				oninput={(e) => (values.nLines = parseInt(e.currentTarget.value) || 0)}
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
				value={values.robust}
				oninput={(e) => (values.robust = parseFloat(e.currentTarget.value) || 0.0)}
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
				value={values.imagingAlgorithm}
				onchange={(e) => (values.imagingAlgorithm = e.currentTarget.value as 'legacy')}
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
				value={values.sourceOffsetXArcsec}
				oninput={(e) => (values.sourceOffsetXArcsec = parseFloat(e.currentTarget.value) || 0.0)}
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
				value={values.sourceOffsetYArcsec}
				oninput={(e) => (values.sourceOffsetYArcsec = parseFloat(e.currentTarget.value) || 0.0)}
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
				value={values.backgroundMode}
				onchange={(e) => (values.backgroundMode = e.currentTarget.value)}
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
				value={values.backgroundLevel}
				oninput={(e) => (values.backgroundLevel = parseFloat(e.currentTarget.value) || 0.0)}
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
				value={values.backgroundSeed ?? ''}
				placeholder="Random"
				oninput={(e) => (values.backgroundSeed = parseOptionalInt(e.currentTarget.value))}
				class={controlClass}
			/>
			<p class="mt-1 text-xs text-fg-dim">Optional reproducible background seed</p>
		</div>
	</div>
</section>

{#if browserOpen}
	<DirectoryPicker
		startPath={values.outputDir || BROWSE_ROOT}
		title="Choose Output Folder"
		allowCreate
		onSelect={selectOutputDir}
		onClose={() => (browserOpen = false)}
	/>
{/if}
