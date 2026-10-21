<!-- A task's quality.json (image clean-quality): each section's figures, rounded. -->
<script lang="ts">
	interface Props {
		quality: Record<string, unknown>;
	}

	let { quality }: Props = $props();

	const sections = $derived(
		Object.entries(quality).filter(
			([, v]) => v !== null && typeof v === 'object' && !Array.isArray(v)
		) as [string, Record<string, unknown>][]
	);

	const label = (key: string) =>
		key
			.replace(/_/g, ' ')
			.replace(/\bjy\b/i, 'Jy')
			.replace(/\bsnr\b/i, 'S/N');

	function show(value: unknown): string {
		if (typeof value === 'number')
			return Number.isInteger(value) ? String(value) : value.toPrecision(3);
		if (Array.isArray(value)) return value.map(show).join(' × ');
		if (value === null || value === undefined) return '—';
		return String(value);
	}
</script>

<div class="flex flex-wrap gap-6">
	{#each sections as [name, figures] (name)}
		<section class="min-w-64 flex-1">
			<h3 class="mt-0 mb-2 text-sm font-semibold capitalize">{name}</h3>
			<dl class="m-0 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-xs">
				{#each Object.entries(figures) as [key, value] (key)}
					{#if value !== null && typeof value === 'object' && !Array.isArray(value)}
						<dt class="col-span-2 mt-1 font-medium text-fg-muted">{label(key)}</dt>
						{#each Object.entries(value as Record<string, unknown>) as [sub, v] (sub)}
							<dt class="pl-3 text-fg-muted">{sub}</dt>
							<dd class="m-0 text-right font-mono">{show(v)}</dd>
						{/each}
					{:else}
						<dt class="text-fg-muted">{label(key)}</dt>
						<dd class="m-0 text-right font-mono">{show(value)}</dd>
					{/if}
				{/each}
			</dl>
		</section>
	{/each}
</div>
