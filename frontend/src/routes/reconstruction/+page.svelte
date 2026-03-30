<script lang="ts">
	import { onMount } from 'svelte';

	onMount(() => {
		if (typeof window === 'undefined') return;
		window.location.replace(`/combination${window.location.search}`);
	});
</script>

<div class="rounded-lg bg-white p-6 shadow-md">
	<h1 class="text-2xl font-bold text-gray-900">Redirecting…</h1>
	<p class="mt-2 text-gray-600">This page has been renamed to Combination.</p>
</div>
