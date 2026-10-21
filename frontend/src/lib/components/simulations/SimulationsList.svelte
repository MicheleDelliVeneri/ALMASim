<script lang="ts">
	import { formatDate } from '#lib/utils/format.js';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import StatusBadge from '#lib/components/shared/StatusBadge.svelte';
	import { isActiveStatus, isTerminalStatus } from '#lib/utils/status.js';
	import { createLogger } from '#lib/logger.js';
	import { isAbortError, ApiTimeoutError } from '#lib/api/client.js';
	import {
		simulationApi,
		type SimulationStatus,
		type SimulationSummary
	} from '#lib/api/simulation.js';

	const logger = createLogger('components/SimulationsList');

	let simulations = $state<SimulationSummary[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let currentRequest = $state<AbortController | null>(null);
	let refreshing = $state(false);
	let logDialogOpen = $state(false);
	let logLoading = $state(false);
	let logError = $state<string | null>(null);
	let selectedLogSimulation = $state<SimulationSummary | null>(null);
	let selectedSimulationStatus = $state<SimulationStatus | null>(null);
	let logSocket: WebSocket | null = null;
	let outputLocation = $state<string | null>(null);
	const SIMULATIONS_FETCH_TIMEOUT_MS = 20_000;
	const SIMULATIONS_REFRESH_INTERVAL_MS = 10_000;

	async function fetchSimulations() {
		if (refreshing) {
			return;
		}

		try {
			refreshing = true;
			if (simulations.length === 0) {
				loading = true;
			}
			error = null;
			currentRequest?.abort();
			const controller = new AbortController();
			currentRequest = controller;
			simulations = await simulationApi.list({
				signal: controller.signal,
				timeoutMs: SIMULATIONS_FETCH_TIMEOUT_MS
			});
			logger.debug({ count: simulations.length }, 'Simulations list refreshed');
		} catch (err) {
			if (err instanceof ApiTimeoutError) {
				error = 'Timed out while loading simulations';
				return;
			}
			if (isAbortError(err)) {
				// Request was superseded or component unmounted; avoid noisy user-facing errors.
				error = null;
				return;
			}
			error = err instanceof Error ? err.message : 'Failed to load simulations';
			logger.error({ err }, 'Error fetching simulations');
		} finally {
			loading = false;
			refreshing = false;
			currentRequest = null;
		}
	}

	function handleVisualize(simulationId: string) {
		const sim = simulations.find((entry) => entry.simulation_id === simulationId);
		const query = sim?.output_dir ? `?dir=${encodeURIComponent(sim.output_dir)}` : '';
		logger.info({ simulationId, outputDir: sim?.output_dir }, 'Navigating to visualizer');
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(resolve('/visualizer') + query);
	}

	function handleCombination(sim: SimulationSummary) {
		const query = sim.output_dir ? `?dir=${encodeURIComponent(sim.output_dir)}` : '';
		logger.info(
			{ simulationId: sim.simulation_id, outputDir: sim.output_dir },
			'Navigating to combination'
		);
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(resolve('/combination') + query);
	}

	function handleImaging(sim: SimulationSummary) {
		const query = sim.output_dir ? `?dir=${encodeURIComponent(sim.output_dir)}` : '';
		logger.info(
			{ simulationId: sim.simulation_id, outputDir: sim.output_dir },
			'Navigating to imaging'
		);
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(resolve('/imaging') + query);
	}

	function handleLocate(sim: SimulationSummary) {
		const path = sim.output_dir
			? `${sim.output_dir}/${sim.simulation_id}`
			: `(output directory not recorded)`;
		logger.info({ simulationId: sim.simulation_id, path }, 'Locate simulation output');
		outputLocation = path;
	}

	function closeLogSocket() {
		if (logSocket) {
			logSocket.close();
			logSocket = null;
		}
	}

	function closeLogDialog() {
		logDialogOpen = false;
		logLoading = false;
		logError = null;
		selectedLogSimulation = null;
		selectedSimulationStatus = null;
		closeLogSocket();
	}

	async function cancelSimulation(sim: SimulationSummary) {
		try {
			await simulationApi.cancel(sim.simulation_id);
			simulations = simulations.map((entry) =>
				entry.simulation_id === sim.simulation_id
					? { ...entry, status: 'cancelled', message: 'Cancellation requested' }
					: entry
			);
			if (selectedLogSimulation?.simulation_id === sim.simulation_id && selectedSimulationStatus) {
				selectedSimulationStatus = {
					...selectedSimulationStatus,
					status: 'cancelled',
					current_step: 'Cancelled',
					message: 'Cancellation requested'
				};
			}
			await fetchSimulations();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to cancel simulation';
			logger.error({ err, simulationId: sim.simulation_id }, 'Failed to cancel simulation');
		}
	}

	function connectLogSocket(simulationId: string) {
		closeLogSocket();
		const socket = simulationApi.openStatusSocket(
			simulationId,
			(status) => {
				selectedSimulationStatus = status;
				logError = null;
			},
			(err) =>
				logger.error({ err, simulationId }, 'Failed to parse simulation log websocket payload')
		);

		socket.onerror = () => {
			logError = 'Log stream disconnected';
		};

		socket.onclose = () => {
			logSocket = null;
		};

		logSocket = socket;
	}

	async function openLogDialog(sim: SimulationSummary) {
		logDialogOpen = true;
		logLoading = true;
		logError = null;
		selectedLogSimulation = sim;
		selectedSimulationStatus = null;
		closeLogSocket();

		try {
			const status = await simulationApi.getStatus(sim.simulation_id);
			selectedSimulationStatus = status;
			if (!isTerminalStatus(status.status)) {
				connectLogSocket(sim.simulation_id);
			}
		} catch (err) {
			logError = err instanceof Error ? err.message : 'Failed to load simulation logs';
			logger.error({ err, simulationId: sim.simulation_id }, 'Failed to load simulation logs');
		} finally {
			logLoading = false;
		}
	}

	onMount(() => {
		logger.info('SimulationsList mounted');
		fetchSimulations();
		// Refresh every 10 seconds to reduce load during long simulations.
		const interval = setInterval(fetchSimulations, SIMULATIONS_REFRESH_INTERVAL_MS);
		return () => {
			clearInterval(interval);
			currentRequest?.abort();
			closeLogSocket();
		};
	});
</script>

<section class="rounded-lg bg-surface p-6 shadow">
	<div class="mb-4 flex items-center justify-between">
		<h2 class="text-xl font-semibold text-fg">Simulations</h2>
		<button
			type="button"
			class="rounded-md border border-line px-3 py-1 text-sm font-medium text-fg-muted hover:bg-hover"
			onclick={fetchSimulations}
		>
			Refresh
		</button>
	</div>

	{#if outputLocation}
		<div
			class="mb-4 flex items-start justify-between gap-3 rounded-md border border-accent-line bg-accent-soft px-4 py-2 text-sm text-fg"
			role="status"
		>
			<span>Simulation output location: <code class="font-mono">{outputLocation}</code></span>
			<button
				type="button"
				class="text-fg-dim hover:text-fg"
				aria-label="Dismiss"
				onclick={() => (outputLocation = null)}>✕</button
			>
		</div>
	{/if}

	{#if loading && simulations.length === 0}
		<div class="flex items-center justify-center py-8">
			<div
				class="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent"
			></div>
		</div>
	{:else if error}
		<div class="rounded-md bg-danger-soft p-4">
			<p class="text-sm text-danger">{error}</p>
		</div>
	{:else if simulations.length === 0}
		<div class="py-8 text-center text-fg-dim">
			<p>No simulations found</p>
			<p class="mt-1 text-sm">Create a simulation to get started</p>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="min-w-full divide-y divide-line">
				<thead class="bg-surface">
					<tr>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							ID
						</th>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							Status
						</th>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							Progress
						</th>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							Created
						</th>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							Updated
						</th>
						<th
							scope="col"
							class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-fg-dim"
						>
							Actions
						</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-line bg-surface">
					{#each simulations as sim (sim.simulation_id)}
						<tr class="hover:bg-hover">
							<td class="whitespace-nowrap px-4 py-3 text-sm font-mono text-fg">
								{sim.simulation_id.slice(0, 8)}...
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm">
								<StatusBadge status={sim.status} />
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-fg">
								<div class="flex items-center gap-2">
									<div class="h-2 w-24 overflow-hidden rounded-full bg-gray-200">
										<div
											class="h-full bg-primary transition-all duration-300"
											style="width: {sim.progress}%"
										></div>
									</div>
									<span class="text-xs text-fg-muted">{Math.round(sim.progress)}%</span>
								</div>
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-fg-dim">
								{formatDate(sim.created_at)}
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-fg-dim">
								{formatDate(sim.updated_at)}
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm">
								<div class="flex gap-2">
									<button
										type="button"
										class="text-accent hover:text-accent disabled:text-gray-400"
										onclick={() => handleVisualize(sim.simulation_id)}
										disabled={sim.status !== 'completed'}
										title="Visualize results"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
											/>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
											/>
										</svg>
									</button>
									<button
										type="button"
										class="text-slate-600 hover:text-slate-800"
										onclick={() => openLogDialog(sim)}
										title="View simulation logs"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M8 10h8M8 14h5M5 5h14a2 2 0 012 2v10a2 2 0 01-2 2H9l-4 3V7a2 2 0 012-2z"
											/>
										</svg>
									</button>
									<button
										type="button"
										class="text-emerald-600 hover:text-emerald-800 disabled:text-gray-400"
										onclick={() => handleCombination(sim)}
										disabled={sim.status !== 'completed'}
										title="Open combination products"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M4 7h16M4 12h10M4 17h7"
											/>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M15 12l2 2 4-4"
											/>
										</svg>
									</button>
									<button
										type="button"
										class="text-violet-600 hover:text-violet-800 disabled:text-gray-400"
										onclick={() => handleImaging(sim)}
										disabled={sim.status !== 'completed'}
										title="Open imaging workflow"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M3 7h18M6 12h12M9 17h6"
											/>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M17 9l3 3-3 3"
											/>
										</svg>
									</button>
									<button
										type="button"
										class="text-fg-muted hover:text-gray-800"
										onclick={() => handleLocate(sim)}
										title="Show file location"
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
									<button
										type="button"
										class="text-danger hover:text-danger disabled:text-gray-400"
										onclick={() => cancelSimulation(sim)}
										disabled={!isActiveStatus(sim.status)}
										title="Cancel simulation"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M6 6l12 12M18 6L6 18"
											/>
										</svg>
									</button>
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</section>

{#if logDialogOpen}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="flex max-h-[85vh] w-full max-w-4xl flex-col rounded-lg bg-surface shadow-panel">
			<header class="flex items-start justify-between gap-4 border-b px-5 py-4">
				<div class="min-w-0">
					<h3 class="text-lg font-semibold text-fg">Simulation Logs</h3>
					{#if selectedLogSimulation}
						<p class="mt-1 truncate font-mono text-xs text-fg-dim">
							{selectedLogSimulation.simulation_id}
						</p>
					{/if}
					{#if selectedSimulationStatus}
						<p class="mt-1 text-sm text-fg-muted">
							Status: <span class="font-medium">{selectedSimulationStatus.status}</span>
							{#if selectedSimulationStatus.current_step}
								| Step: <span class="font-medium">{selectedSimulationStatus.current_step}</span>
							{/if}
						</p>
					{/if}
				</div>
				<button
					type="button"
					class="rounded-md border border-line px-3 py-1.5 text-sm font-medium text-fg-muted hover:bg-hover"
					onclick={closeLogDialog}
				>
					Close
				</button>
			</header>

			<div class="flex-1 space-y-3 overflow-hidden px-5 py-4">
				{#if logLoading}
					<div class="rounded-md bg-surface px-4 py-3 text-sm text-fg-muted">Loading logs…</div>
				{/if}

				{#if logError}
					<div class="rounded-md bg-danger-soft px-4 py-3 text-sm text-danger">{logError}</div>
				{/if}

				<div class="overflow-hidden rounded-md border border-line bg-slate-950">
					<div class="max-h-[55vh] overflow-y-auto p-4 font-mono text-xs leading-5 text-slate-100">
						{#if selectedSimulationStatus?.logs && selectedSimulationStatus.logs.length > 0}
							{#each selectedSimulationStatus.logs as line, i (i)}
								<div class="whitespace-pre-wrap break-words">{line}</div>
							{/each}
						{:else if !logLoading}
							<div class="text-fg-dim">No logs recorded yet.</div>
						{/if}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}
