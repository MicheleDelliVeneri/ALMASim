<script lang="ts">
	import { onMount } from 'svelte';

	interface SimulationStatus {
		simulation_id: string;
		status: string;
		progress: number;
		current_step: string;
		message: string;
		logs: string[];
		error?: string;
	}

	interface Props {
		simulationId: string;
		status: SimulationStatus | null;
	}

	let { simulationId, status }: Props = $props();

	let logsContainer = $state<HTMLDivElement>();

	const SIMULATION_STEPS = [
		'Initializing',
		'Generating antenna configuration',
		'Computing max baseline',
		'Creating sky model',
		'Running interferometric simulation',
		'Processing results',
		'Saving output'
	];

	// Auto-scroll logs to bottom when they update
	$effect(() => {
		if (logsContainer && status?.logs) {
			setTimeout(() => {
				if (logsContainer) {
					logsContainer.scrollTop = logsContainer.scrollHeight;
				}
			}, 0);
		}
	});
</script>

<section class="space-y-4 rounded-lg bg-white p-6 shadow-md">
	<div class="flex items-center justify-between">
		<h3 class="text-lg font-semibold text-gray-900">Simulation Status</h3>
		{#if status}
			<span
				class={`rounded-full px-3 py-1 text-xs font-medium ${
					status?.status === 'completed'
						? 'bg-green-100 text-green-800'
						: status?.status === 'failed'
							? 'bg-red-100 text-red-800'
							: status?.status === 'running'
								? 'bg-blue-100 text-blue-800'
								: 'bg-gray-100 text-gray-800'
				}`}
			>
				{status?.status.toUpperCase() || 'CONNECTING'}
			</span>
		{/if}
	</div>

	<div class="text-sm text-gray-600">
		<p>
			ID: <code class="rounded bg-gray-100 px-2 py-1 text-xs">{simulationId}</code>
		</p>
		{#if status}
			<p class="mt-2">
				Current Step: <span class="font-medium">{status?.current_step || 'N/A'}</span>
			</p>
			<p class="mt-1">Message: <span class="text-gray-800">{status?.message || 'N/A'}</span></p>
		{/if}
	</div>

	{#if status}
		<div class="space-y-2">
			<div class="flex items-center justify-between text-sm">
				<span class="text-gray-700">Progress</span>
				<span class="font-medium text-gray-900">{Math.round(status?.progress || 0)}%</span>
			</div>
			<div class="h-3 w-full overflow-hidden rounded-full bg-gray-200">
				<div
					class="h-full bg-blue-600 transition-all duration-300 ease-out"
					style="width: {status?.progress || 0}%"
				></div>
			</div>
		</div>

		<div class="space-y-2">
			<p class="text-sm font-medium text-gray-700">Simulation Steps:</p>
			<div class="space-y-1">
				{#each SIMULATION_STEPS as step, index}
					{@const currentStep = status?.current_step || ''}
					{@const stepProgress = ((index + 1) / SIMULATION_STEPS.length) * 100}
					{@const isActive =
						currentStep.toLowerCase().includes(step.toLowerCase()) ||
						(status?.progress || 0) >= stepProgress}
					{@const isCompleted = (status?.progress || 0) > stepProgress}

					<div class="flex items-center space-x-2 text-sm">
						<div
							class={`flex h-4 w-4 items-center justify-center rounded-full ${
								isCompleted ? 'bg-green-500' : isActive ? 'bg-blue-500' : 'bg-gray-300'
							}`}
						>
							{#if isCompleted}
								<span class="text-xs text-white">✓</span>
							{/if}
						</div>
						<span class={isActive ? 'font-medium text-gray-900' : 'text-gray-500'}>
							{step}
						</span>
					</div>
				{/each}
			</div>
		</div>

		{#if status?.error}
			<div class="rounded-md border border-red-200 bg-red-50 p-3">
				<p class="text-sm font-medium text-red-800">Error:</p>
				<p class="mt-1 text-sm text-red-700">{status?.error}</p>
			</div>
		{/if}

		<div class="rounded-md border border-gray-300 bg-gray-900 font-mono text-xs text-gray-100">
			<div class="flex items-center justify-between rounded-t-md bg-gray-800 px-3 py-2">
				<span class="font-medium text-gray-300">Backend Logs</span>
				<span class="text-xs text-gray-500">{status?.logs?.length || 0} entries</span>
			</div>
			<div bind:this={logsContainer} class="max-h-64 space-y-1 overflow-y-auto p-3">
				{#if status?.logs && status.logs.length > 0}
					{#each status.logs as log}
						<div class="whitespace-pre-wrap break-words text-gray-300">
							{log}
						</div>
					{/each}
				{:else}
					<p class="text-gray-500">No logs yet...</p>
				{/if}
			</div>
		</div>
	{:else}
		<div class="py-4 text-center text-gray-500">
			<p>Connecting to simulation status...</p>
		</div>
	{/if}
</section>
