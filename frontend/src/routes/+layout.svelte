<script lang="ts">
	import '../app.css';

	const navItems = [
		{ href: '/', label: 'Overview' },
		{ href: '/simulations', label: 'Simulations' },
		{ href: '/metadata', label: 'Metadata' },
		{ href: '/visualizer', label: 'Visualizer' }
	];
</script>

<div class="flex min-h-screen bg-gray-50 text-gray-900">
	<aside class="hidden w-64 flex-col bg-slate-900 text-white shadow-lg md:flex">
		<div class="px-6 py-5 text-2xl font-semibold tracking-tight">ALMASim</div>
		<nav class="flex-1 px-4 py-4">
			{#each navItems as item}
				<a
					href={item.href}
					class="mb-2 block rounded-md px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 hover:text-white"
					data-sveltekit-preload-data
				>
					{item.label}
				</a>
			{/each}
		</nav>
		<div class="px-6 py-4 text-xs text-slate-400">© {new Date().getFullYear()} ALMASim</div>
	</aside>
	<main class="flex-1 overflow-y-auto">
		<div class="mx-auto max-w-6xl px-4 py-8">
			<slot />
		</div>
	</main>
</div>
