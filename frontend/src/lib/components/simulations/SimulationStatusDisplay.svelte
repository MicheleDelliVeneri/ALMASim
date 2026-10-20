<script lang="ts">
	import { createLogger } from '#lib/logger.js';

	const logger = createLogger('components/simulations/SimulationStatusDisplay');

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
		onCancel?: () => void | Promise<void>;
	}

	let { simulationId, status, onCancel }: Props = $props();

	let logsContainer = $state<HTMLDivElement>();

	const SIMULATION_STEPS = [
		'Initializing',
		'Generating clean cube',
		'Running interferometric simulation',
		'Reconstructing image products',
		'Exporting results'
	];

	// Log meaningful status transitions
	let _lastStatus = '';
	$effect(() => {
		if (status && status.status !== _lastStatus) {
			_lastStatus = status.status;
			if (status.status === 'completed') {
				logger.info({ simulationId, progress: status.progress }, 'Simulation completed');
			} else if (status.status === 'failed') {
				logger.error({ simulationId, error: status.error }, 'Simulation failed');
			} else if (status.status === 'running') {
				logger.info({ simulationId }, 'Simulation running');
			}
		}
	});

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

<section class="space-y-4 rounded-lg bg-surface p-6 shadow-panel">
	<div class="flex items-center justify-between">
		<h3 class="text-lg font-semibold text-fg">Simulation Status</h3>
		{#if status}
			<div class="flex items-center gap-3">
				<span
					class={`rounded-full px-3 py-1 text-xs font-medium ${
						status?.status === 'completed'
							? 'bg-ok-soft text-ok'
							: status?.status === 'failed'
								? 'bg-danger-soft text-danger'
								: status?.status === 'running'
									? 'bg-accent-soft text-accent'
									: status?.status === 'cancelled'
										? 'bg-surface text-fg'
										: 'bg-surface text-fg'
					}`}
				>
					{status?.status.toUpperCase() || 'CONNECTING'}
				</span>
				{#if onCancel && (status.status === 'queued' || status.status === 'running')}
					<button
						type="button"
						class="rounded-md border border-danger-line px-3 py-1 text-xs font-medium text-danger hover:bg-danger-soft"
						onclick={() => onCancel?.()}
					>
						Cancel
					</button>
				{/if}
			</div>
		{/if}
	</div>

	<div class="text-sm text-fg-muted">
		<p>
			ID: <code class="rounded bg-surface px-2 py-1 text-xs">{simulationId}</code>
		</p>
		{#if status}
			<p class="mt-2">
				Current Step: <span class="font-medium">{status?.current_step || 'N/A'}</span>
			</p>
			<p class="mt-1">Message: <span class="text-fg">{status?.message || 'N/A'}</span></p>
		{/if}
	</div>

	{#if status}
		<div class="space-y-2">
			<div class="flex items-center justify-between text-sm">
				<span class="text-fg-muted">Progress</span>
				<span class="font-medium text-fg">{Math.round(status?.progress || 0)}%</span>
			</div>
			<div class="h-3 w-full overflow-hidden rounded-full bg-gray-200">
				<div
					class="h-full bg-primary transition-all duration-300 ease-out"
					style="width: {status?.progress || 0}%"
				></div>
			</div>
		</div>

		<div class="space-y-2">
			<p class="text-sm font-medium text-fg-muted">Simulation Steps:</p>
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
						<span class={isActive ? 'font-medium text-fg' : 'text-fg-dim'}>
							{step}
						</span>
					</div>
				{/each}
			</div>
		</div>

		{#if status?.error}
			<div class="rounded-md border border-danger-line bg-danger-soft p-3">
				<p class="text-sm font-medium text-danger">Error:</p>
				<p class="mt-1 text-sm text-danger">{status?.error}</p>
			</div>
		{/if}

		<div class="rounded-md border border-line bg-gray-900 font-mono text-xs text-gray-100">
			<div class="flex items-center justify-between rounded-t-md bg-gray-800 px-3 py-2">
				<span class="font-medium text-gray-300">Backend Logs</span>
				<span class="text-xs text-fg-dim">{status?.logs?.length || 0} entries</span>
			</div>
			<div bind:this={logsContainer} class="max-h-64 space-y-1 overflow-y-auto p-3">
				{#if status?.logs && status.logs.length > 0}
					{#each status.logs as log}
						<div class="whitespace-pre-wrap break-words text-gray-300">
							{log}
						</div>
					{/each}
				{:else}
					<p class="text-fg-dim">No logs yet...</p>
				{/if}
			</div>
		</div>
	{:else}
		<div class="py-4 text-center text-fg-dim">
			<p>Connecting to simulation status...</p>
		</div>
	{/if}
</section>
