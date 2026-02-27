<script lang="ts">
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

	let fileInputRef: HTMLInputElement;

	function handleBrowseDirectory() {
		fileInputRef.click();
	}

	function handleDirectorySelect(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			// Get the path of the first file and extract the directory
			const file = input.files[0];
			const path = (file as any).path || file.webkitRelativePath || file.name;
			const directory = path.substring(0, path.lastIndexOf('/')) || '/';
			onOutputDirChange(directory);
		}
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
					placeholder="/app/outputs"
					class="flex-1 rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
				/>
				<button
					type="button"
					onclick={handleBrowseDirectory}
					class="rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
					title="Browse for directory"
				>
					<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
					</svg>
				</button>
				<input
					bind:this={fileInputRef}
					type="file"
					webkitdirectory
					directory
					multiple
					onchange={handleDirectorySelect}
					class="hidden"
				/>
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
