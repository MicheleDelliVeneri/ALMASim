<script lang="ts">
	import { onMount } from 'svelte';
	import { apiClient } from '$lib/api/client';

	let healthStatus = $state('checking...');

	onMount(async () => {
		try {
			const response = await apiClient.get<{ status: string }>('/health');
			healthStatus = response.status === 'healthy' ? '✅ Healthy' : '❌ Unhealthy';
		} catch (error) {
			healthStatus = '❌ Error connecting to API';
		}
	});
</script>

<div class="container mx-auto px-4 py-8">
	<div class="mx-auto max-w-4xl">
		<header class="mb-8">
			<h1 class="mb-2 text-4xl font-bold text-gray-900">ALMASim</h1>
			<p class="text-lg text-gray-600">
				Generate realistic ALMA observations with advanced simulation capabilities
			</p>
			<div class="mt-4">
				<span class="inline-block rounded-full bg-blue-100 px-3 py-1 text-sm text-blue-800">
					API Status: {healthStatus}
				</span>
			</div>
		</header>

		<div class="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
			<a
				href="/simulations"
				class="block rounded-lg bg-white p-6 shadow-md transition-shadow hover:shadow-lg"
			>
				<h2 class="mb-2 text-xl font-semibold text-gray-900">Simulations</h2>
				<p class="text-gray-600">Create and manage ALMA simulations</p>
			</a>

			<a
				href="/metadata"
				class="block rounded-lg bg-white p-6 shadow-md transition-shadow hover:shadow-lg"
			>
				<h2 class="mb-2 text-xl font-semibold text-gray-900">Metadata</h2>
				<p class="text-gray-600">Query and browse ALMA observation metadata</p>
			</a>
		</div>
	</div>
</div>
