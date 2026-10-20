<script lang="ts">
	import { downloadApi } from '#lib/api/download.js';
	import { createLogger } from '#lib/logger.js';
	import { formatBytes } from '#lib/utils/format.js';

	const logger = createLogger('components/metadata/SelectionSizeEstimate');

	interface Props {
		/** The selected member OUS UIDs; a new selection discards the last estimate. */
		uids: string[];
		/** The selected rows, shown on the button. */
		count: number;
	}

	let { uids, count }: Props = $props();

	let resolving = $state(false);
	let resolved = $state(0); // UIDs resolved so far
	let total = $state(0); // UIDs to resolve
	let totalBytes = $state(0);
	let totalFiles = $state(0);
	let error = $state('');
	let cancelled = $state(false);

	function reset() {
		resolved = 0;
		total = 0;
		totalBytes = 0;
		totalFiles = 0;
		error = '';
		cancelled = false;
	}

	$effect(() => {
		void uids;
		reset();
	});

	const finished = $derived(!resolving && resolved > 0);

	/** Resolve the products of each UID in turn, adding up their sizes. */
	async function compute() {
		const todo = [...uids];
		if (todo.length === 0) return;
		reset();
		resolving = true;
		total = todo.length;
		try {
			for (let i = 0; i < todo.length; i++) {
				if (cancelled) {
					logger.info({ resolved: i, total: todo.length }, 'Size computation cancelled');
					break;
				}
				const result = await downloadApi.resolveProducts([todo[i]]);
				totalBytes += result.products.reduce(
					(sum, product) => sum + (product.content_length || 0),
					0
				);
				totalFiles += result.products.length;
				resolved = i + 1;
			}
			if (!cancelled) {
				logger.info({ totalSize: formatBytes(totalBytes), files: totalFiles }, 'Size computed');
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to compute size';
			logger.error({ err: e }, 'Failed to compute size');
		} finally {
			resolving = false;
		}
	}
</script>

{#if resolving}
	<span
		class="inline-flex items-center gap-2 rounded-md border border-accent-line bg-accent-soft px-3 py-2 text-sm font-medium text-accent"
	>
		<svg class="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
			<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25" />
			<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
		</svg>
		{resolved}/{total}
		{#if totalBytes > 0}
			— {formatBytes(totalBytes)}
		{/if}
	</span>
	<button
		type="button"
		class="rounded-md border border-danger-line px-3 py-2 text-sm font-medium text-danger hover:bg-danger-soft"
		onclick={() => (cancelled = true)}
		title="Stop computing size"
	>
		Stop
	</button>
{:else}
	<button
		type="button"
		class="rounded-md border border-accent-line px-4 py-2 text-sm font-medium text-accent hover:bg-accent-soft"
		onclick={compute}
		title="Compute total download size for selected observations"
	>
		Compute Size ({count})
	</button>
{/if}
{#if finished}
	<span
		class="inline-flex items-center gap-1.5 rounded-md border border-accent-line bg-accent-soft px-3 py-2 text-sm font-medium text-accent"
	>
		{formatBytes(totalBytes)} — {totalFiles} files
		{#if cancelled}
			<span class="text-xs text-amber-600">({resolved}/{total} resolved)</span>
		{/if}
	</span>
{:else if error}
	<span
		class="inline-flex items-center gap-1.5 rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger"
	>
		{error}
		{#if resolved > 0}
			({resolved}/{total} resolved — {formatBytes(totalBytes)})
		{/if}
	</span>
{/if}
