<script lang="ts">
	import '@fontsource-variable/noto-sans';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import type { Snippet } from 'svelte';
	import '../app.css';

	let { children }: { children: Snippet } = $props();

	const navItems = [
		{ href: '/', label: 'Overview' },
		{ href: '/metadata', label: 'Metadata' },
		{ href: '/data', label: 'Data' },
		{ href: '/simulations', label: 'Simulations' },
		{ href: '/combination', label: 'Combination' },
		{ href: '/imaging', label: 'Imaging' },
		{ href: '/visualizer', label: 'Visualizer' }
	] as const;

	function current(href: string): boolean {
		const path = page.url.pathname;
		return href === '/' ? path === '/' : path === href || path.startsWith(`${href}/`);
	}
</script>

<div class="flex min-h-screen flex-col bg-canvas text-fg">
	<header class="sticky top-0 z-40 bg-navy text-white">
		<div class="flex h-16 items-center gap-4 px-4 sm:px-6">
			<a href={resolve('/')} class="flex items-center gap-3 text-white no-underline">
				<img src="/assets/skao-mark-white.png" alt="" class="h-8 w-8" />
				<span class="text-xl font-light tracking-tight">ALMASim</span>
				<span class="hidden text-xs font-medium uppercase tracking-wide text-navy-soft sm:inline"
					>Web console</span
				>
			</a>
		</div>
		<nav
			aria-label="Main"
			class="flex gap-1 overflow-x-auto border-t border-white/10 px-2 py-2 md:hidden"
		>
			{#each navItems as item (item.href)}
				<a
					href={resolve(item.href)}
					aria-current={current(item.href) ? 'page' : undefined}
					class="shrink-0 rounded px-3 py-1.5 text-sm text-navy-soft aria-[current=page]:bg-white/15 aria-[current=page]:text-white"
					data-sveltekit-preload-data>{item.label}</a
				>
			{/each}
		</nav>
	</header>

	<div class="flex min-h-0 flex-1">
		<nav
			aria-label="Main"
			class="hidden w-60 shrink-0 flex-col gap-0.5 border-r border-line bg-surface px-3 py-4 md:flex"
		>
			{#each navItems as item (item.href)}
				<a
					href={resolve(item.href)}
					aria-current={current(item.href) ? 'page' : undefined}
					class="rounded px-3 py-2 text-sm font-medium text-fg-muted hover:bg-hover hover:text-navy aria-[current=page]:bg-sunk aria-[current=page]:font-bold aria-[current=page]:text-navy"
					data-sveltekit-preload-data>{item.label}</a
				>
			{/each}
			<div class="mt-auto px-3 pt-6 text-xs text-fg-dim">© {new Date().getFullYear()} ALMASim</div>
		</nav>

		<main class="min-w-0 flex-1 overflow-y-auto">
			<div class="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
				{@render children()}
			</div>
		</main>
	</div>
</div>
