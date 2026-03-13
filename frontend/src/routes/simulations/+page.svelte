<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { simulationApi, type SimulationParamsCreate } from '$lib/api/simulation';
	import type { MetadataResponse } from '$lib/api/metadata';
	import ComputeBackendConfig from '$lib/components/simulations/ComputeBackendConfig.svelte';
	import SimulationForm from '$lib/components/simulations/SimulationForm.svelte';
	import MetadataSelector from '$lib/components/simulations/MetadataSelector.svelte';
	import SelectedRowPreview from '$lib/components/simulations/SelectedRowPreview.svelte';
	import SimulationStatusDisplay from '$lib/components/simulations/SimulationStatusDisplay.svelte';
	import SimulationsList from '$lib/components/simulations/SimulationsList.svelte';
	import { createLogger } from '$lib/logger';

	const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
	const logger = createLogger('routes/simulations');

	interface SimulationStatus {
		simulation_id: string;
		status: string;
		progress: number;
		current_step: string;
		message: string;
		logs: string[];
		error?: string;
	}

	const RESULTS_CACHE_KEY = 'almasim:metadata-results';

	const getCachedResults = (): MetadataResponse | null => {
		if (typeof window === 'undefined') return null;
		try {
			const raw = window.localStorage.getItem(RESULTS_CACHE_KEY);
			return raw ? (JSON.parse(raw) as MetadataResponse) : null;
		} catch {
			return null;
		}
	};

	let loading = $state(false);
	let message = $state('');
	let simulationId = $state('');
	let metadata = $state<MetadataResponse | null>(null);
	let selectedRowIndices = $state<number[]>([]);

	let simulationStatus = $state<SimulationStatus | null>(null);
	let ws: WebSocket | null = null;

	// Simulation form state
	let nPix = $state(256);
	let nChannels = $state(128);
	let sourceType = $state('point');
	let simulationName = $state('');
	let outputDir = $state('/host_home');
	let snr = $state(1.3);
	let saveMode = $state('npz');
	let nLines = $state(0);
	let robust = $state(0.0);
	let numSimulations = $state(1);

	let computeBackend = $state('local');
	let backendConfig = $state<Record<string, unknown>>({});

	onMount(() => {
		logger.info('Simulations page mounted');
		const cached = getCachedResults();
		if (cached) {
			metadata = cached;
			logger.debug({ rowCount: cached.data.length }, 'Loaded cached metadata');
		} else {
			logger.debug('No cached metadata found');
		}
	});

	onDestroy(() => {
		if (ws) {
			ws.close();
		}
	});

	function connectWebSocket(id: string) {
		const wsUrl = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
		const socket = new WebSocket(`${wsUrl}/api/v1/simulations/${id}/ws`);

		socket.onopen = () => {
			logger.info({ simulationId: id }, 'WebSocket connected');
		};

		socket.onmessage = (event) => {
			try {
				const status: SimulationStatus = JSON.parse(event.data);
				simulationStatus = status;
			} catch (e) {
				logger.error({ err: e, simulationId: id }, 'Failed to parse WebSocket message');
			}
		};

		socket.onerror = (error) => {
			logger.error({ err: error, simulationId: id }, 'WebSocket error');
		};

		socket.onclose = () => {
			logger.info({ simulationId: id }, 'WebSocket disconnected');
			ws = null;
		};

		ws = socket;
	}

	const selectedRow = $derived.by(() => {
		const indices = selectedRowIndices;
		const data = metadata?.data;
		if (indices.length === 0 || !data) return null;
		const idx = indices[0]; // Use first selected row for preview
		if (idx < 0 || idx >= data.length) return null;
		return data[idx];
	});

	function getRowValue(row: Record<string, unknown>, key: string): string {
		const value = row[key];
		if (value === null || value === undefined || value === '') return 'N/A';
		if (typeof value === 'number') {
			if (isNaN(value)) return 'N/A';
			return value.toString();
		}
		if (Array.isArray(value)) return value.join(', ');
		if (typeof value === 'object') return JSON.stringify(value);
		return String(value);
	}

	function getRowNumber(row: Record<string, unknown>, key: string): number | null {
		const value = row[key];
		if (value === null || value === undefined || value === '') return null;
		if (typeof value === 'number') {
			if (isNaN(value)) return null;
			return value;
		}
		if (typeof value === 'string') {
			const parsed = parseFloat(value);
			return isNaN(parsed) ? null : parsed;
		}
		return null;
	}

	async function handleSubmit(event: Event) {
		event.preventDefault();
		loading = true;
		message = '';

		logger.info(
			{ sourceType, nPix, nChannels, numSimulations, selectedRowCount: selectedRowIndices.length },
			'Simulation form submitted'
		);

		// Get selected rows or use first selected row
		const data = metadata?.data;
		if (!data || selectedRowIndices.length === 0) {
			message = 'Error: Please select at least one metadata row first.';
			loading = false;
			logger.warn('Simulation submit blocked: no metadata row selected');
			return;
		}

		// Determine rows to process
		const rowsToProcess: Array<Record<string, unknown>> = [];

		if (numSimulations > 1 && selectedRowIndices.length === 1) {
			// Create multiple simulations from the same row
			const row = data[selectedRowIndices[0]];
			for (let i = 0; i < numSimulations; i++) {
				rowsToProcess.push(row);
			}
		} else {
			// Create one simulation per selected row
			for (const idx of selectedRowIndices) {
				if (idx >= 0 && idx < data.length) {
					rowsToProcess.push(data[idx]);
				}
			}
		}

		if (rowsToProcess.length === 0) {
			message = 'Error: No valid rows to process.';
			loading = false;
			return;
		}

		// Process all rows
		const createdSimulations: string[] = [];
		const errors: string[] = [];

		for (let i = 0; i < rowsToProcess.length; i++) {
			const row = rowsToProcess[i];

			const getValue = (keys: string[], fallback: any = null) => {
				for (const key of keys) {
					const value = row[key];
					if (value !== null && value !== undefined && value !== '') {
						return value;
					}
				}
				return fallback;
			};

			const getNumber = (keys: string[], fallback: number): number => {
				const value = getValue(keys, fallback);
				if (typeof value === 'number') {
					if (isNaN(value)) return fallback;
					return value;
				}
				if (typeof value === 'string') {
					const parsed = parseFloat(value);
					return isNaN(parsed) ? fallback : parsed;
				}
				return fallback;
			};

			const getString = (keys: string[], fallback: string): string => {
				const value = getValue(keys, fallback);
				return String(value ?? fallback);
			};

			const simParams: SimulationParamsCreate = {
				idx: i,
				source_name: getString(['ALMA_source_name', 'source_name'], 'Unknown'),
				member_ouid: getString(['member_ous_uid', 'member_ouid'], 'unknown'),
				project_name: simulationName || getString(['proposal_id', 'project_name'], 'ALMASim'),
				ra: getNumber(['RA', 'ra'], 0.0),
				dec: getNumber(['Dec', 'dec'], 0.0),
				band: getNumber(['Band', 'band'], 3),
				ang_res: getNumber(['Ang.res.', 'Ang.res', 'ang_res'], 0.1),
				vel_res: getNumber(['Vel.res.', 'Vel.res', 'vel_res'], 1.0),
				fov: getNumber(['FOV', 'fov'], 10.0),
				obs_date: getString(['Obs.date', 'obs_date'], new Date().toISOString().split('T')[0]),
				pwv: getNumber(['PWV', 'pwv'], 1.0),
				int_time: getNumber(['Int.Time', 'int_time'], 3600.0),
				bandwidth: getNumber(['Bandwidth', 'bandwidth'], 2.0),
				freq: getNumber(['Freq', 'freq'], 100.0),
				freq_support: getString(['Freq.sup.', 'Freq.sup', 'freq_support'], '100.0-102.0'),
				cont_sens: getNumber(['Cont_sens_mJybeam', 'Cont_sens', 'cont_sens'], 0.1),
				antenna_array: getString(['antenna_arrays', 'antenna_array'], 'C43-1'),
				source_type: sourceType,
				n_pix: nPix,
				n_channels: nChannels,
				snr: snr,
				save_mode: saveMode,
				n_lines: nLines > 0 ? nLines : undefined,
				robust: robust,
				main_dir: './src/almasim',
				output_dir: outputDir || './outputs',
				tng_dir: './data/TNG100-1',
				galaxy_zoo_dir: './data/galaxy_zoo',
				hubble_dir: './data/hubble',
				compute_backend: computeBackend,
				compute_backend_config: backendConfig
			};

			try {
				const response = await simulationApi.create(simParams);
				createdSimulations.push(response.simulation_id);
				logger.info({ simulationId: response.simulation_id, index: i }, 'Simulation created');

				// Connect to first simulation's websocket
				if (i === 0) {
					simulationId = response.simulation_id;
					connectWebSocket(response.simulation_id);
				}
			} catch (error) {
				errors.push(
					`Simulation ${i + 1}: ${error instanceof Error ? error.message : 'Unknown error'}`
				);
				logger.error({ err: error, index: i }, 'Failed to create simulation');
			}
		}

		// Show results
		loading = false;

		if (createdSimulations.length > 0 && errors.length === 0) {
			message = `Successfully created ${createdSimulations.length} simulation(s)!`;
			logger.info({ count: createdSimulations.length, ids: createdSimulations }, 'All simulations created successfully');
		} else if (createdSimulations.length > 0 && errors.length > 0) {
			message = `Created ${createdSimulations.length} simulation(s) with ${errors.length} error(s). Errors: ${errors.join('; ')}`;
			logger.warn({ created: createdSimulations.length, failed: errors.length }, 'Some simulations failed to create');
		} else {
			message = `Error: Failed to create simulations. ${errors.join('; ')}`;
			logger.error({ errors }, 'All simulations failed to create');
		}
	}
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-6xl space-y-6">
		<header>
			<h1 class="text-3xl font-bold text-gray-900">Create Simulation</h1>
			<p class="mt-2 text-gray-600">Select a metadata row and configure simulation parameters.</p>
		</header>

		{#if message}
			<div
				class={`rounded-lg p-4 ${
					message.includes('Error')
						? 'border border-red-200 bg-red-100 text-red-800'
						: 'border border-green-200 bg-green-100 text-green-800'
				}`}
			>
				{message}
			</div>
		{/if}

		<ComputeBackendConfig
			backendType={computeBackend}
			{backendConfig}
			onBackendTypeChange={(type) => (computeBackend = type)}
			onConfigChange={(config) => (backendConfig = config)}
		/>

		<SimulationForm
			{sourceType}
			{nPix}
			{nChannels}
			{simulationName}
			{outputDir}
			{snr}
			{saveMode}
			{nLines}
			{robust}
			{numSimulations}
			onSourceTypeChange={(type) => (sourceType = type)}
			onNPixChange={(value) => (nPix = value)}
			onNChannelsChange={(value) => (nChannels = value)}
			onSimulationNameChange={(value) => (simulationName = value)}
			onOutputDirChange={(value) => (outputDir = value)}
			onSnrChange={(value) => (snr = value)}
			onSaveModeChange={(value) => (saveMode = value)}
			onNLinesChange={(value) => (nLines = value)}
			onRobustChange={(value) => (robust = value)}
			onNumSimulationsChange={(value) => (numSimulations = value)}
		/>

		<MetadataSelector
			{metadata}
			selectedIndices={selectedRowIndices}
			onSelect={(indices) => (selectedRowIndices = indices)}
			{getRowValue}
			{getRowNumber}
		/>

		<SelectedRowPreview
			row={selectedRow}
			{getRowValue}
			{getRowNumber}
			{sourceType}
			{nPix}
			{nChannels}
		/>

		<form onsubmit={handleSubmit}>
			<button
				type="submit"
				disabled={loading || !selectedRow}
				class="w-full rounded-md bg-blue-600 px-6 py-3 text-lg font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-400"
			>
				{loading ? 'Creating Simulation...' : 'Create Simulation'}
			</button>
			<p class="mt-2 text-center text-xs text-gray-500">
				{!selectedRow
					? 'Please select a metadata row above'
					: `Will create ${sourceType} simulation with ${nPix}×${nPix}×${nChannels} cube`}
			</p>
		</form>

		{#if simulationId}
			<SimulationStatusDisplay {simulationId} status={simulationStatus} />
		{/if}

		<div class="mt-8">
			<SimulationsList apiUrl={API_BASE_URL} />
		</div>
	</div>
</div>
