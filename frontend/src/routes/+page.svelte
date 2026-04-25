<script lang="ts">
	import { onMount } from 'svelte';
	import { apiClient } from '$lib/api/client';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('routes/home');

	let healthStatus = $state('checking');
	let healthTone = $state('warn');

	const modules = [
		{
			href: '/metadata',
			code: '01',
			title: 'Metadata',
			description: 'Query ALMA observations and select real datasets for simulation or data processing.',
			metric: 'TAP'
		},
		{
			href: '/data',
			code: '02',
			title: 'Data',
			description: 'Download ALMA products, unpack raw MeasurementSets, and generate calibrated visibilities.',
			metric: 'MS'
		},
		{
			href: '/simulations',
			code: '03',
			title: 'Simulations',
			description: 'Configure and run ALMA simulations from selected metadata and source models.',
			metric: 'SIM'
		},
		{
			href: '/visualizer',
			code: '04',
			title: 'Visualizer',
			description: 'Inspect simulation outputs, image products, datacubes, and reconstructed data.',
			metric: 'VIS'
		}
	];

	const workflowLinks = [
		{ href: '/metadata', label: 'Find observations' },
		{ href: '/data', label: 'Download products' },
		{ href: '/simulations', label: 'Run simulations' },
		{ href: '/imaging', label: 'Create images' }
	];

	onMount(async () => {
		logger.info('Home page mounted');
		try {
			const response = await apiClient.get<{ status: string }>('/health');
			healthStatus = response.status === 'healthy' ? 'connected' : 'degraded';
			healthTone = response.status === 'healthy' ? 'secondary' : 'secondary';
			logger.info({ status: response.status }, 'API health check complete');
		} catch (error) {
			healthStatus = 'offline';
			healthTone = 'error';
			logger.error({ err: error }, 'API health check failed');
		}
	});
</script>

<div class="space-y-8">
	<section class="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
		<div class="alma-panel rounded-lg p-6 sm:p-8">
			<div class="alma-kicker">ALMA Simulator</div>
			<div class="mt-5 max-w-3xl">
				<h1 class="text-4xl font-bold leading-tight text-gray-100 sm:text-5xl">
					ALMASim
				</h1>
				<p class="mt-4 text-lg leading-8 text-gray-300">
					A control surface for ALMA simulation workflows: select observations, download data products,
					create MeasurementSets, run simulations, image outputs, and inspect results.
				</p>
			</div>
			<div class="mt-8 flex flex-wrap gap-3">
				<a href="/metadata" class="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700">
					Start from Metadata
				</a>
				<a href="/simulations" class="rounded border border-gray-300 px-4 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-50">
					Open Simulations
				</a>
			</div>
		</div>

		<div class="alma-panel rounded-lg p-6">
			<div class="border-b border-gray-200 pb-4">
				<div class="flex items-start justify-between gap-4">
					<div class="min-w-0">
						<div class="alma-kicker">Connection</div>
						<div class="mt-2 text-2xl font-semibold text-gray-100">Backend</div>
					</div>
					<span
						class="shrink-0 rounded px-3 py-1 text-xs font-semibold uppercase tracking-wide"
						class:alma-chip={healthTone === 'secondary'}
						class:bg-red-100={healthTone === 'error'}
						class:text-red-700={healthTone === 'error'}
					>
						{healthStatus}
					</span>
				</div>
			</div>
			<div class="mt-5 space-y-3">
				<p class="text-sm leading-6 text-gray-400">
					This only reports whether the frontend can reach the ALMASim backend API.
				</p>
				<div class="grid gap-2">
					{#each workflowLinks as link}
						<a href={link.href} class="flex items-center justify-between rounded border border-gray-200 px-3 py-2 text-sm text-gray-300 hover:bg-gray-50">
							<span>{link.label}</span>
							<span class="text-gray-500">Open</span>
						</a>
					{/each}
				</div>
			</div>
		</div>
	</section>

	<section class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
		{#each modules as module}
			<a href={module.href} class="alma-panel group rounded-lg p-5 transition hover:border-cyan-300/50 hover:bg-white/10">
				<div class="flex items-center justify-between">
					<span class="font-mono text-xs text-gray-500">{module.code}</span>
					<span class="alma-chip rounded px-2 py-1 text-xs font-semibold">{module.metric}</span>
				</div>
				<h2 class="mt-5 text-xl font-semibold text-gray-100">{module.title}</h2>
				<p class="mt-3 text-sm leading-6 text-gray-400">{module.description}</p>
				<div class="mt-5 text-sm font-semibold text-blue-600 group-hover:text-cyan-200">Open module</div>
			</a>
		{/each}
	</section>
</div>
