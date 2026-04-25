<script lang="ts">
	import '../app.css';

	const navItems = [
		{ href: '/', label: 'Overview', code: 'OVR' },
		{ href: '/metadata', label: 'Metadata', code: 'META' },
		{ href: '/data', label: 'Data', code: 'DATA' },
		{ href: '/simulations', label: 'Simulations', code: 'SIM' },
		{ href: '/combination', label: 'Combination', code: 'COMB' },
		{ href: '/imaging', label: 'Imaging', code: 'IMG' },
		{ href: '/visualizer', label: 'Visualizer', code: 'VIS' }
	];
</script>

<div class="alma-shell min-h-screen text-gray-100">
	<div class="flex min-h-screen">
		<aside class="alma-glass hidden w-72 shrink-0 flex-col border-r border-gray-200/20 md:flex">
			<div class="border-b border-gray-200/20 px-6 py-6">
				<div class="alma-kicker">ALMA Simulator</div>
				<div class="mt-3 flex items-center gap-3">
					<img
						src="/assets/almasim-icon.jpeg"
						alt="ALMASim icon"
						class="h-11 w-11 rounded border border-blue-300/30 object-cover"
					/>
					<div>
						<div class="text-2xl font-bold tracking-tight text-gray-100">ALMASim</div>
						<div class="text-xs text-gray-400">A simulator for a more civilized age </div>
					</div>
				</div>
			</div>

			<nav class="flex-1 px-4 py-5">
				{#each navItems as item}
					<a
						href={item.href}
						class="group mb-2 flex items-center gap-3 rounded px-3 py-2.5 text-sm font-medium text-gray-300 transition hover:bg-white/10 hover:text-white"
						data-sveltekit-preload-data
					>
						<span class="w-10 rounded border border-gray-200/20 bg-black/20 px-2 py-1 text-center font-mono text-[10px] text-cyan-200 group-hover:border-cyan-200/50">
							{item.code}
						</span>
						<span>{item.label}</span>
					</a>
				{/each}
			</nav>

			<div class="border-t border-gray-200/20 px-6 py-5">
				<div class="alma-chip rounded px-3 py-2 text-xs">
					<div class="font-semibold uppercase tracking-wide">Runtime</div>
					<div class="mt-1 text-gray-300">Simulation and data services</div>
				</div>
				<div class="mt-4 text-xs text-gray-500">© {new Date().getFullYear()} ALMASim</div>
			</div>
		</aside>

		<div class="flex min-w-0 flex-1 flex-col">
			<header class="alma-glass sticky top-0 z-40 border-b border-gray-200/20 md:hidden">
				<div class="flex items-center justify-between px-4 py-3">
					<div class="flex items-center gap-3">
						<img
							src="/assets/almasim-icon.jpeg"
							alt="ALMASim icon"
							class="h-9 w-9 rounded border border-blue-300/30 object-cover"
						/>
						<div>
							<div class="alma-kicker">ALMASim</div>
							<div class="text-sm text-gray-300">ALMA Simulator</div>
						</div>
					</div>
					<a href="/data" class="alma-chip rounded px-3 py-1 text-xs font-semibold">DATA</a>
				</div>
				<div class="flex gap-2 overflow-x-auto px-4 pb-3">
					{#each navItems as item}
						<a href={item.href} class="rounded border border-gray-200/20 px-3 py-1.5 text-xs text-gray-300" data-sveltekit-preload-data>
							{item.label}
						</a>
					{/each}
				</div>
			</header>

			<main class="min-w-0 flex-1 overflow-y-auto">
				<div class="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
					<slot />
				</div>
			</main>
		</div>
	</div>
</div>
