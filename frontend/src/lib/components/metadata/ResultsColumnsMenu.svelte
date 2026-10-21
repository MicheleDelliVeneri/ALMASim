<script lang="ts">
	import { createLogger } from '#lib/logger.js';
	import { PINNED_COLUMNS, formatColumnName, moveColumn } from '#lib/utils/resultsTable.js';

	const logger = createLogger('components/metadata/ResultsColumnsMenu');

	interface Props {
		/** The reorderable columns, in display order. */
		columnOrder: string[];
		hiddenColumns: Set<string>;
		/** Columns present in the current results; the others are greyed out. */
		dataColumns: Set<string>;
		visibleCount: number;
	}

	let {
		columnOrder = $bindable(),
		hiddenColumns = $bindable(),
		dataColumns,
		visibleCount
	}: Props = $props();

	let open = $state(false);
	let dragged = $state<string | null>(null);
	let dragOver = $state<string | null>(null);

	function toggleColumn(col: string) {
		const next = new Set(hiddenColumns);
		if (next.has(col)) {
			next.delete(col);
			logger.debug({ col }, 'Column shown');
		} else {
			next.add(col);
			logger.debug({ col }, 'Column hidden');
		}
		hiddenColumns = next;
	}

	function onDragStart(col: string, e: DragEvent) {
		dragged = col;
		e.dataTransfer!.effectAllowed = 'move';
		e.dataTransfer!.setData('text/plain', col);
	}

	function onDragOver(col: string, e: DragEvent) {
		if (!dragged) return;
		e.preventDefault();
		dragOver = col;
	}

	function onDrop(col: string) {
		if (dragged) columnOrder = moveColumn(columnOrder, dragged, col);
		dragged = null;
		dragOver = null;
	}

	function onDragEnd() {
		dragged = null;
		dragOver = null;
	}
</script>

<svelte:window
	onclick={(e) => {
		if (!(e.target as HTMLElement).closest('[data-columns-menu]')) open = false;
	}}
/>

<div class="relative" data-columns-menu>
	<button
		type="button"
		class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover"
		onclick={() => (open = !open)}
	>
		Columns ({visibleCount}/{PINNED_COLUMNS.length + columnOrder.length})
	</button>
	{#if open}
		<div
			class="absolute right-0 z-30 mt-1 max-h-96 w-64 overflow-y-auto rounded-md border border-line bg-surface shadow-panel"
		>
			<div class="flex items-center justify-between border-b border-line px-3 py-2">
				<span class="text-xs font-semibold uppercase tracking-wide text-fg-dim">Show / Hide</span>
				<div class="flex gap-2">
					<button
						type="button"
						class="text-xs text-accent hover:underline"
						onclick={() => (hiddenColumns = new Set())}>All</button
					>
					<button
						type="button"
						class="text-xs text-accent hover:underline"
						onclick={() => (hiddenColumns = new Set(columnOrder))}>None</button
					>
				</div>
			</div>
			<!-- Pinned columns (always visible, not reorderable) -->
			{#each PINNED_COLUMNS as col (col)}
				<div class="flex items-center gap-2 bg-surface px-3 py-1.5 text-sm text-fg-dim">
					<svg class="h-3.5 w-3.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
						<path
							fill-rule="evenodd"
							d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
							clip-rule="evenodd"
						/>
					</svg>
					{formatColumnName(col)}
					<span class="ml-auto text-xs text-gray-300">pinned</span>
				</div>
			{/each}
			<!-- Reorderable columns (drag up/down to reorder) -->
			{#each columnOrder as col (col)}
				<label
					class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm text-fg-muted transition-colors hover:bg-hover"
					class:text-fg-dim={!dataColumns.has(col)}
					class:border-t-2={dragOver === col && dragged !== col}
					class:border-blue-400={dragOver === col && dragged !== col}
					draggable="true"
					ondragstart={(e: DragEvent) => onDragStart(col, e)}
					ondragover={(e: DragEvent) => onDragOver(col, e)}
					ondrop={() => onDrop(col)}
					ondragend={onDragEnd}
					role="listitem"
				>
					<span
						class="cursor-grab select-none text-gray-300 hover:text-gray-500"
						aria-label="Drag to reorder">⠇⠇</span
					>
					<input
						type="checkbox"
						checked={!hiddenColumns.has(col)}
						onchange={() => toggleColumn(col)}
						class="rounded border-line"
					/>
					{formatColumnName(col)}
					{#if !dataColumns.has(col)}
						<span class="ml-auto text-xs text-gray-300">—</span>
					{/if}
				</label>
			{/each}
		</div>
	{/if}
</div>
