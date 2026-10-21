<script lang="ts">
	import { BROWSE_ROOT } from '#lib/constants.js';
	import { readCachedResults } from '#lib/utils/metadataCache.js';
	import { onMount, onDestroy } from 'svelte';
	import {
		simulationApi,
		type SimulationEstimate,
		type SimulationParamsCreate
	} from '#lib/api/simulation.js';
	import type { MetadataResponse } from '#lib/api/metadata.js';
	import ComputeBackendConfig from '#lib/components/simulations/ComputeBackendConfig.svelte';
	import SimulationForm from '#lib/components/simulations/SimulationForm.svelte';
	import { defaultSimulationForm, formRequestFields } from '#lib/utils/simulationForm.js';
	import MetadataSelector from '#lib/components/simulations/MetadataSelector.svelte';
	import SelectedRowPreview from '#lib/components/simulations/SelectedRowPreview.svelte';
	import SimulationStatusDisplay from '#lib/components/simulations/SimulationStatusDisplay.svelte';
	import SimulationsList from '#lib/components/simulations/SimulationsList.svelte';
	import { createLogger } from '#lib/logger.js';
	import { inferObservationConfigsFromMetadataRow } from '#lib/utils/observationPlan.js';

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

	let loading = $state(false);
	let message = $state('');
	let simulationId = $state('');
	let metadata = $state<MetadataResponse | null>(null);
	let selectedRowIndices = $state<number[]>([]);

	let simulationStatus = $state<SimulationStatus | null>(null);
	let simulationEstimate = $state<SimulationEstimate | null>(null);
	let estimating = $state(false);
	let estimateError = $state<string | null>(null);
	let ws: WebSocket | null = null;

	// Simulation form state
	let form = $state(defaultSimulationForm(BROWSE_ROOT));

	let computeBackend = $state('local');
	let backendConfig = $state<Record<string, unknown>>({});

	onMount(() => {
		logger.info('Simulations page mounted');
		const cached = readCachedResults();
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
		if (ws) {
			ws.close();
			ws = null;
		}
		const socket = simulationApi.openStatusSocket(
			id,
			(status) => {
				simulationStatus = status as SimulationStatus;
			},
			(e) => logger.error({ err: e, simulationId: id }, 'Failed to parse WebSocket message')
		);

		socket.onopen = () => {
			logger.info({ simulationId: id }, 'WebSocket connected');
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

	function getOptionalNumber(row: Record<string, unknown>, keys: string[]): number | undefined {
		for (const key of keys) {
			const value = getRowNumber(row, key);
			if (value !== null) return value;
		}
		return undefined;
	}

	async function cancelActiveSimulation() {
		if (!simulationId || !simulationStatus) return;
		try {
			const response = await simulationApi.cancel(simulationId);
			simulationStatus = {
				...(simulationStatus ?? {
					simulation_id: simulationId,
					progress: 0,
					current_step: 'Cancelled',
					message: '',
					logs: []
				}),
				status: response.status,
				current_step: 'Cancelled',
				message: response.message || 'Cancellation requested'
			};
			message = response.message || 'Cancellation requested';
		} catch (error) {
			message = `Error: ${error instanceof Error ? error.message : 'Failed to cancel simulation'}`;
		}
	}

	function buildSimulationParamsFromRow(
		row: Record<string, unknown>,
		idx: number
	): SimulationParamsCreate {
		const getValue = (keys: string[], fallback: unknown = null) => {
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

		return {
			idx,
			source_name: getString(['ALMA_source_name', 'source_name'], 'Unknown'),
			member_ouid: getString(['member_ous_uid', 'member_ouid'], 'unknown'),
			project_name: form.simulationName || getString(['proposal_id', 'project_name'], 'ALMASim'),
			ra: getNumber(['RA', 'ra'], 0.0),
			dec: getNumber(['Dec', 'dec'], 0.0),
			band: getNumber(['Band', 'band'], 3),
			ang_res: getNumber(['Ang.res.', 'Ang.res', 'ang_res'], 0.1),
			vel_res: getNumber(['Vel.res.', 'Vel.res', 'vel_res'], 1.0),
			fov: getNumber(['FOV', 'fov'], 10.0),
			obs_date: getString(['Obs.date', 'obs_date'], new Date().toISOString().split('T')[0]),
			pwv: form.useMetadataPwv ? getNumber(['PWV', 'pwv'], 1.0) : form.pwvOverride,
			int_time: getNumber(['Int.Time', 'int_time'], 3600.0),
			bandwidth: getNumber(['Bandwidth', 'bandwidth'], 2.0),
			freq: getNumber(['Freq', 'freq'], 100.0),
			freq_support: getString(['Freq.sup.', 'Freq.sup', 'freq_support'], '100.0-102.0'),
			cont_sens: getNumber(['Cont_sens_mJybeam', 'Cont_sens', 'cont_sens'], 0.1),
			line_sens_10kms: getOptionalNumber(row, ['Line_sens_10kms_mJybeam']),
			antenna_array: getString(['antenna_arrays', 'antenna_array'], 'C43-1'),
			observation_configs: inferObservationConfigsFromMetadataRow(
				row,
				getNumber(['Int.Time', 'int_time'], 3600.0)
			),
			...formRequestFields(form),
			compute_backend: computeBackend,
			compute_backend_config: backendConfig
		};
	}

	$effect(() => {
		const row = selectedRow;
		if (!row) {
			simulationEstimate = null;
			estimateError = null;
			return;
		}

		// Read the form now, so this effect reruns when it changes, but ask for
		// the estimate only once typing pauses: every keystroke used to POST.
		const params = buildSimulationParamsFromRow(row, 0);
		let cancelled = false;
		const controller = new AbortController();
		estimating = true;
		estimateError = null;

		const timer = setTimeout(() => {
			void simulationApi
				.estimate(params, { signal: controller.signal })
				.then((estimate) => {
					if (!cancelled) {
						simulationEstimate = estimate;
					}
				})
				.catch((error) => {
					if (!cancelled) {
						simulationEstimate = null;
						estimateError =
							error instanceof Error ? error.message : 'Failed to estimate simulation size';
					}
				})
				.finally(() => {
					if (!cancelled) {
						estimating = false;
					}
				});
		}, 500);

		return () => {
			cancelled = true;
			clearTimeout(timer);
			controller.abort();
		};
	});

	async function handleSubmit(event: Event) {
		event.preventDefault();
		loading = true;
		message = '';
		simulationStatus = null;
		if (ws) {
			ws.close();
			ws = null;
		}

		logger.info(
			{
				sourceType: form.sourceType,
				nPix: form.nPix,
				nChannels: form.nChannels,
				numSimulations: form.numSimulations,
				selectedRowCount: selectedRowIndices.length
			},
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

		if (form.numSimulations > 1 && selectedRowIndices.length === 1) {
			// Create multiple simulations from the same row
			const row = data[selectedRowIndices[0]];
			for (let i = 0; i < form.numSimulations; i++) {
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

			const simParams = buildSimulationParamsFromRow(row, i);

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
			logger.info(
				{ count: createdSimulations.length, ids: createdSimulations },
				'All simulations created successfully'
			);
		} else if (createdSimulations.length > 0 && errors.length > 0) {
			message = `Created ${createdSimulations.length} simulation(s) with ${errors.length} error(s). Errors: ${errors.join('; ')}`;
			logger.warn(
				{ created: createdSimulations.length, failed: errors.length },
				'Some simulations failed to create'
			);
		} else {
			message = `Error: Failed to create simulations. ${errors.join('; ')}`;
			logger.error({ errors }, 'All simulations failed to create');
		}
	}
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-6xl space-y-6">
		<header>
			<h1 class="text-3xl font-bold text-fg">Create Simulation</h1>
			<p class="mt-2 text-fg-muted">Select a metadata row and configure simulation parameters.</p>
		</header>

		{#if message}
			<div
				class={`rounded-lg p-4 ${
					message.includes('Error')
						? 'border border-danger-line bg-danger-soft text-danger'
						: 'border border-ok-line bg-ok-soft text-ok'
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

		<SimulationForm bind:values={form} />

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
			{form}
			estimate={simulationEstimate}
			{estimating}
			{estimateError}
		/>

		<form onsubmit={handleSubmit}>
			<button
				type="submit"
				disabled={loading || !selectedRow}
				class="w-full rounded-md bg-primary px-6 py-3 text-lg font-medium text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:bg-gray-400"
			>
				{loading ? 'Creating Simulation...' : 'Create Simulation'}
			</button>
			<p class="mt-2 text-center text-xs text-fg-dim">
				{!selectedRow
					? 'Please select a metadata row above'
					: form.nPix !== null && form.nChannels !== null
						? `Will create ${form.sourceType} simulation with ${form.nPix}×${form.nPix}×${form.nChannels} cube`
						: 'Will auto-compute cube dimensions from the selected metadata row unless you set overrides above'}
			</p>
		</form>

		{#if simulationId}
			<SimulationStatusDisplay
				{simulationId}
				status={simulationStatus}
				onCancel={cancelActiveSimulation}
			/>
		{/if}

		<div class="mt-8">
			<SimulationsList />
		</div>
	</div>
</div>
