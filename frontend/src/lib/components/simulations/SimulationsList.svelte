<script lang="ts">
	import { onMount } from 'svelte';
	import { createLogger } from '$lib/logger';

	interface SimulationSummary {
		simulation_id: string;
		status: string;
		progress: number;
		message: string;
		created_at: string;
		updated_at: string;
		error?: string;
		output_dir?: string;
	}

	interface Props {
		apiUrl?: string;
	}

	let { apiUrl = 'http://localhost:8000' }: Props = $props();
	const logger = createLogger('components/SimulationsList');

	let simulations = $state<SimulationSummary[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	async function fetchSimulations() {
		try {
			loading = true;
			error = null;
			const response = await fetch(`${apiUrl}/api/v1/simulations/`);
			if (!response.ok) {
				throw new Error(`Failed to fetch simulations: ${response.statusText}`);
			}
			const data = await response.json();
			simulations = data.simulations || [];
			logger.debug({ count: simulations.length }, 'Simulations list refreshed');
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load simulations';
			logger.error({ err, apiUrl }, 'Error fetching simulations');
		} finally {
			loading = false;
		}
	}

	function getStatusColor(status: string): string {
		switch (status) {
			case 'completed':
				return 'bg-green-100 text-green-800';
			case 'running':
				return 'bg-blue-100 text-blue-800';
			case 'failed':
				return 'bg-red-100 text-red-800';
			case 'queued':
				return 'bg-yellow-100 text-yellow-800';
			default:
				return 'bg-gray-100 text-gray-800';
		}
	}

	function formatDate(dateString: string): string {
		const date = new Date(dateString);
		return date.toLocaleString();
	}

	function handleVisualize(simulationId: string) {
		logger.info({ simulationId }, 'Navigating to visualizer');
		// Navigate to visualizer with simulation output
		window.location.href = `/visualizer?simulation=${simulationId}`;
	}

	function handleLocate(sim: SimulationSummary) {
		const path = sim.output_dir
			? `${sim.output_dir}/${sim.simulation_id}`
			: `(output directory not recorded)`;
		logger.info({ simulationId: sim.simulation_id, path }, 'Locate simulation output');
		alert(`Simulation output location:\n${path}`);
	}

	onMount(() => {
		logger.info('SimulationsList mounted');
		fetchSimulations();
		// Refresh every 5 seconds
		const interval = setInterval(fetchSimulations, 5000);
		return () => clearInterval(interval);
	});
</script>

<section class="rounded-lg bg-white p-6 shadow">
	<div class="mb-4 flex items-center justify-between">
		<h2 class="text-xl font-semibold text-gray-900">Simulations</h2>
		<button
			type="button"
			class="rounded-md border border-gray-300 px-3 py-1 text-sm font-medium text-gray-700 hover:bg-gray-50"
			onclick={fetchSimulations}
		>
			Refresh
		</button>
	</div>

	{#if loading && simulations.length === 0}
		<div class="flex items-center justify-center py-8">
			<div class="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent"></div>
		</div>
	{:else if error}
		<div class="rounded-md bg-red-50 p-4">
			<p class="text-sm text-red-800">{error}</p>
		</div>
	{:else if simulations.length === 0}
		<div class="py-8 text-center text-gray-500">
			<p>No simulations found</p>
			<p class="mt-1 text-sm">Create a simulation to get started</p>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="min-w-full divide-y divide-gray-200">
				<thead class="bg-gray-50">
					<tr>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							ID
						</th>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							Status
						</th>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							Progress
						</th>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							Created
						</th>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							Updated
						</th>
						<th scope="col" class="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
							Actions
						</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100 bg-white">
					{#each simulations as sim}
						<tr class="hover:bg-gray-50">
							<td class="whitespace-nowrap px-4 py-3 text-sm font-mono text-gray-900">
								{sim.simulation_id.slice(0, 8)}...
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm">
								<span class="inline-flex rounded-full px-2 py-1 text-xs font-semibold {getStatusColor(sim.status)}">
									{sim.status}
								</span>
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-gray-900">
								<div class="flex items-center gap-2">
									<div class="h-2 w-24 overflow-hidden rounded-full bg-gray-200">
										<div
											class="h-full bg-blue-600 transition-all duration-300"
											style="width: {sim.progress}%"
										></div>
									</div>
									<span class="text-xs text-gray-600">{Math.round(sim.progress)}%</span>
								</div>
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-gray-500">
								{formatDate(sim.created_at)}
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm text-gray-500">
								{formatDate(sim.updated_at)}
							</td>
							<td class="whitespace-nowrap px-4 py-3 text-sm">
								<div class="flex gap-2">
									<button
										type="button"
										class="text-blue-600 hover:text-blue-800 disabled:text-gray-400"
										onclick={() => handleVisualize(sim.simulation_id)}
										disabled={sim.status !== 'completed'}
										title="Visualize results"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
										</svg>
									</button>
									<button
										type="button"
										class="text-gray-600 hover:text-gray-800"
										onclick={() => handleLocate(sim)}
										title="Show file location"
									>
										<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
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
