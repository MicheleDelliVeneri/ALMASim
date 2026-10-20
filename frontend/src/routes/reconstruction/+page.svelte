<script lang="ts">
	import { onMount } from 'svelte';

	onMount(() => {
		if (typeof window === 'undefined') return;
		window.location.replace(`/combination${window.location.search}`);
	});
</script>

<div class="rounded-lg bg-surface p-6 shadow-panel">
	<h1 class="text-2xl font-bold text-fg">Redirecting…</h1>
	<p class="mt-2 text-fg-muted">This page has been renamed to Combination.</p>
</div>
