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

	const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
	let selectedRowIndex = $state<number | null>(null);

	let simulationStatus = $state<SimulationStatus | null>(null);
	let ws: WebSocket | null = null;

	let nPix = $state(256);
	let nChannels = $state(128);
	let sourceType = $state('point');
	let computeBackend = $state('local');
	let backendConfig = $state<Record<string, unknown>>({});

	onMount(() => {
		const cached = getCachedResults();
		if (cached) {
			metadata = cached;
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
			console.log('WebSocket connected');
		};

		socket.onmessage = (event) => {
			try {
				const status: SimulationStatus = JSON.parse(event.data);
				simulationStatus = status;
			} catch (e) {
				console.error('Failed to parse WebSocket message:', e);
			}
		};

		socket.onerror = (error) => {
			console.error('WebSocket error:', error);
		};

		socket.onclose = () => {
			console.log('WebSocket disconnected');
			ws = null;
		};

		ws = socket;
	}

	const selectedRow = $derived.by(() => {
		const idx = selectedRowIndex;
		const data = metadata?.data;
		if (idx === null || !data || idx < 0 || idx >= data.length) return null;
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

		const row = selectedRow;
		if (!row) {
			message = 'Error: Please select a metadata row first.';
			loading = false;
			return;
		}

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

		const params: SimulationParamsCreate = {
			idx: 0,
			source_name: getString(['ALMA_source_name', 'source_name'], 'Unknown'),
			member_ouid: getString(['member_ous_uid', 'member_ouid'], 'unknown'),
			project_name: getString(['proposal_id', 'project_name'], 'ALMASim'),
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
			main_dir: './src/almasim',
			output_dir: './outputs',
			tng_dir: './data/TNG100-1',
			galaxy_zoo_dir: './data/galaxy_zoo',
			hubble_dir: './data/hubble',
			compute_backend: computeBackend,
			compute_backend_config: backendConfig
		};

		try {
			const response = await simulationApi.create(params);
			simulationId = response.simulation_id;
			connectWebSocket(response.simulation_id);
			message = `Simulation created successfully! ID: ${response.simulation_id}`;
		} catch (error) {
			message = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
		} finally {
			loading = false;
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
			onSourceTypeChange={(type) => (sourceType = type)}
			onNPixChange={(value) => (nPix = value)}
			onNChannelsChange={(value) => (nChannels = value)}
		/>

		<MetadataSelector
			{metadata}
			selectedIndices={selectedRowIndex !== null ? [selectedRowIndex] : []}
			onSelect={(indices) => (selectedRowIndex = indices[0] ?? null)}
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
