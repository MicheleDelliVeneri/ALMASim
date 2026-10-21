<!--
	A query's results: choose and order the columns, filter each column (a
	substring, or a glob with *), and tick rows. Only the rows in view are
	drawn, so thousands of rows stay fast.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import ResultsColumnsMenu from './ResultsColumnsMenu.svelte';
	import {
		DEFAULT_COLUMN_ORDER,
		DEFAULT_HIDDEN,
		PINNED_COLUMNS,
		filterRowIndices,
		formatColumnName,
		virtualWindow
	} from '#lib/utils/resultsTable.js';

	interface Props {
		records: Record<string, string>[];
		/** The columns the CSV has. */
		columns: string[];
		/** Positions in `records` of the ticked rows. */
		selected: Set<number>;
	}

	let { records, columns, selected = $bindable() }: Props = $props();

	const ROW_HEIGHT = 36;
	const VIEW_HEIGHT = 480;

	const known = new Set([...PINNED_COLUMNS, ...DEFAULT_COLUMN_ORDER]);
	// The order starts from the first result set; the page remounts the table for a new one.
	let columnOrder = $state([
		...DEFAULT_COLUMN_ORDER,
		...untrack(() => columns).filter((c) => !known.has(c))
	]);
	let hiddenColumns = $state(new Set(DEFAULT_HIDDEN));
	let filters = $state<Record<string, string>>({});
	let scrollTop = $state(0);

	const dataColumns = $derived(new Set(columns));
	const visible = $derived([
		...PINNED_COLUMNS.filter((c) => dataColumns.has(c)),
		...columnOrder.filter((c) => dataColumns.has(c) && !hiddenColumns.has(c))
	]);
	const shownRows = $derived(filterRowIndices(records, filters));
	const windowRows = $derived(
		virtualWindow(scrollTop, shownRows.length, ROW_HEIGHT, VIEW_HEIGHT, 10)
	);
	const allShownTicked = $derived(shownRows.length > 0 && shownRows.every((i) => selected.has(i)));

	/** A cell as shown: long decimals to six significant digits (the full value is its title). */
	function show(value: string): string {
		return /^-?\d+\.\d{6,}$/.test(value) ? String(Number(Number(value).toPrecision(6))) : value;
	}

	function toggle(i: number) {
		const next = new Set(selected);
		if (next.has(i)) next.delete(i);
		else next.add(i);
		selected = next;
	}

	function toggleShown() {
		const next = new Set(selected);
		if (allShownTicked) for (const i of shownRows) next.delete(i);
		else for (const i of shownRows) next.add(i);
		selected = next;
	}
</script>

<div class="flex flex-col gap-3">
	<div class="flex flex-wrap items-center gap-3">
		<p class="m-0 text-sm text-fg-muted" aria-live="polite">
			{shownRows.length.toLocaleString()} of {records.length.toLocaleString()} rows ·
			<strong class="text-navy">{selected.size.toLocaleString()} ticked</strong>
		</p>
		<span class="flex-1"></span>
		{#if selected.size}
			<button type="button" class="btn-ghost px-3 py-1.5" onclick={() => (selected = new Set())}
				>Clear ticks</button
			>
		{/if}
		<ResultsColumnsMenu
			bind:columnOrder
			bind:hiddenColumns
			{dataColumns}
			visibleCount={visible.length}
		/>
	</div>

	<div
		class="overflow-auto rounded-md border border-line"
		style="max-height: {VIEW_HEIGHT + 2 * ROW_HEIGHT}px"
		onscroll={(e) => (scrollTop = Math.max(0, e.currentTarget.scrollTop - 2 * ROW_HEIGHT))}
	>
		<table class="w-full border-collapse text-sm">
			<thead class="sticky top-0 z-10">
				<tr>
					<th scope="col" class="w-10 border-b border-line-strong px-3 py-2">
						<input
							type="checkbox"
							aria-label="Tick every row shown"
							checked={allShownTicked}
							onchange={toggleShown}
						/>
					</th>
					{#each visible as column (column)}
						<th
							scope="col"
							class="border-b border-line-strong px-3 py-2 text-left text-xs font-semibold tracking-wide whitespace-nowrap text-fg-muted uppercase"
							>{formatColumnName(column)}</th
						>
					{/each}
				</tr>
				<tr>
					<td class="border-b border-line bg-sunk"></td>
					{#each visible as column (column)}
						<td class="border-b border-line bg-sunk px-2 py-1">
							<input
								class="w-full min-w-24 rounded border px-2 py-1 text-xs"
								aria-label="Filter {formatColumnName(column)}"
								placeholder="filter…"
								value={filters[column] ?? ''}
								oninput={(e) => (filters = { ...filters, [column]: e.currentTarget.value })}
							/>
						</td>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#if windowRows.topPad}<tr style="height: {windowRows.topPad}px"><td></td></tr>{/if}
				{#each shownRows.slice(windowRows.start, windowRows.end) as i (i)}
					<tr
						class="border-b border-sunk hover:bg-canvas {selected.has(i) ? 'bg-accent-soft' : ''}"
						style="height: {ROW_HEIGHT}px"
					>
						<td class="px-3">
							<input
								type="checkbox"
								aria-label="Tick row {i + 1}"
								checked={selected.has(i)}
								onchange={() => toggle(i)}
							/>
						</td>
						{#each visible as column (column)}
							<td class="max-w-xs truncate px-3 whitespace-nowrap" title={records[i][column]}
								>{show(records[i][column])}</td
							>
						{/each}
					</tr>
				{/each}
				{#if windowRows.bottomPad}<tr style="height: {windowRows.bottomPad}px"><td></td></tr>{/if}
			</tbody>
		</table>
	</div>
</div>
