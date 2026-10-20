<script lang="ts">
	import type { MetadataResponse, MetadataRow } from '#lib/api/metadata.js';
	import { createLogger } from '#lib/logger.js';
	import {
		DEFAULT_COLUMN_ORDER,
		DEFAULT_HIDDEN,
		PINNED_COLUMNS,
		filterRowIndices,
		formatColumnName,
		moveColumn,
		stringifyValue,
		virtualWindow
	} from '#lib/utils/resultsTable.js';
	import ResultsColumnsMenu from './ResultsColumnsMenu.svelte';
	import SelectionSizeEstimate from './SelectionSizeEstimate.svelte';

	const logger = createLogger('components/metadata/MetadataResultsTable');

	interface Props {
		results: MetadataResponse | null;
		/** Identifies the result set: filters and selection reset only when it changes. */
		resultsKey?: unknown;
		loading: boolean;
		fetching: boolean;
		onClear: () => void;
		onLoad: () => void;
		onSave: () => void;
		saving: boolean;
		onDownload: (memberOusUids: string[], metadataRows: MetadataRow[]) => void;
	}

	let { results, resultsKey, loading, fetching, onClear, onLoad, onSave, saving, onDownload }: Props =
		$props();

	let isCollapsed = $state(false);

	// The reorderable columns' order, and the hidden ones.
	let columnOrder = $state([...DEFAULT_COLUMN_ORDER]);
	let hiddenColumns = $state(new Set<string>(DEFAULT_HIDDEN));

	// Client-side column filters (reset on new results)
	let columnFilters = $state<Record<string, string>>({});

	const hasActiveFilters = $derived(Object.values(columnFilters).some((v) => v.length > 0));

	// Columns that actually appear in the current result data
	const dataColumnSet = $derived.by(() => {
		const data = results?.data;
		if (!data || data.length === 0) return new Set<string>();
		return new Set(Object.keys(data[0]));
	});

	const pinnedTableColumns = $derived(PINNED_COLUMNS.filter((c) => dataColumnSet.has(c)));
	const reorderableTableColumns = $derived(
		columnOrder.filter((c) => !hiddenColumns.has(c) && dataColumnSet.has(c))
	);
	const tableColumns = $derived([...pinnedTableColumns, ...reorderableTableColumns]);

	// Positions in results.data of the rows that pass the column filters.
	const filteredIndices = $derived(filterRowIndices(results?.data ?? [], columnFilters));
	const filteredData = $derived(filteredIndices.map((index) => (results?.data ?? [])[index]));

	// --- Row selection ---
	// Positions in results.data, so a selection survives filter changes: an index
	// into the filtered rows would silently point at other rows once the filter
	// changed, and the download would then use the wrong member_ous_uids.
	let selectedRowIndices = $state(new Set<number>());

	// Reset filters and selection when the result set changes, not when the same
	// result set gets more rows (a streaming query rebuilds `results` on every poll).
	let lastResultsKey: unknown = undefined;
	$effect(() => {
		const key = resultsKey !== undefined ? resultsKey : results;
		if (key !== lastResultsKey) {
			lastResultsKey = key;
			columnFilters = {};
			selectedRowIndices = new Set();
		}
	});

	const selectedCount = $derived(selectedRowIndices.size);

	const allFilteredSelected = $derived(
		filteredIndices.length > 0 && filteredIndices.every((i) => selectedRowIndices.has(i))
	);

	function toggleRowSelection(dataIndex: number) {
		const next = new Set(selectedRowIndices);
		if (next.has(dataIndex)) {
			next.delete(dataIndex);
		} else {
			next.add(dataIndex);
		}
		selectedRowIndices = next;
	}

	function toggleSelectAll() {
		if (allFilteredSelected) {
			selectedRowIndices = new Set();
			logger.debug('All rows deselected');
		} else {
			selectedRowIndices = new Set(filteredIndices);
			logger.debug({ count: filteredData.length }, 'All filtered rows selected');
		}
	}

	// The selected rows' distinct member OUS UIDs.
	const selectedMemberOusUids = $derived.by(() => {
		const data = results?.data ?? [];
		const uids: string[] = [];
		for (const idx of selectedRowIndices) {
			const row = data[idx];
			if (row?.member_ous_uid) uids.push(String(row.member_ous_uid));
		}
		return [...new Set(uids)];
	});

	function download() {
		const rows = [...selectedRowIndices]
			.map((idx) => results?.data?.[idx])
			.filter((row): row is MetadataRow => Boolean(row))
			.map((row) => ({ ...row }));
		logger.info(
			{ selectedRows: selectedRowIndices.size, uniqueUids: selectedMemberOusUids.length },
			'Download initiated from table'
		);
		onDownload(selectedMemberOusUids, rows);
	}

	// Virtual scrolling
	const ROW_HEIGHT = 36; // px — matches row height below
	const OVERSCAN = 15; // extra rows above/below viewport
	const CONTAINER_HEIGHT = 480; // must match max-height in CSS

	// Mirror-scroll: top scrollbar synced to the real scroll container
	let scrollContainer = $state<HTMLDivElement | null>(null);
	let mirrorBar = $state<HTMLDivElement | null>(null);
	let tableScrollWidth = $state(0);
	let scrollTop = $state(0);

	const rowWindow = $derived(
		virtualWindow(scrollTop, filteredData.length, ROW_HEIGHT, CONTAINER_HEIGHT, OVERSCAN)
	);
	const visibleRows = $derived(filteredData.slice(rowWindow.start, rowWindow.end));

	$effect(() => {
		if (!scrollContainer || !mirrorBar) return;

		const obs = new ResizeObserver(() => {
			tableScrollWidth = scrollContainer!.scrollWidth;
		});
		obs.observe(scrollContainer);

		const onScroll = () => {
			mirrorBar!.scrollLeft = scrollContainer!.scrollLeft;
			scrollTop = scrollContainer!.scrollTop;
		};
		const onMirrorScroll = () => {
			scrollContainer!.scrollLeft = mirrorBar!.scrollLeft;
		};

		scrollContainer.addEventListener('scroll', onScroll, { passive: true });
		mirrorBar.addEventListener('scroll', onMirrorScroll);

		return () => {
			obs.disconnect();
			scrollContainer!.removeEventListener('scroll', onScroll);
			mirrorBar!.removeEventListener('scroll', onMirrorScroll);
		};
	});

	let pinnedColEls = $state<HTMLElement[]>([]);
	let pinnedColWidths = $state<number[]>([0, 0]);

	$effect(() => {
		const els = pinnedColEls.filter(Boolean);
		if (els.length === 0) return;
		const obs = new ResizeObserver(() => {
			pinnedColWidths = els.map((el) => el.offsetWidth);
		});
		els.forEach((el) => obs.observe(el));
		return () => obs.disconnect();
	});

	const CHECKBOX_COL_WIDTH = 40;

	function pinnedLeft(index: number): number {
		let left = CHECKBOX_COL_WIDTH; // offset for checkbox column
		for (let i = 0; i < index; i++) left += pinnedColWidths[i];
		return left;
	}

	// --- Drag-and-drop: table column headers ---
	let draggedTableCol = $state<string | null>(null);
	let dragOverTableCol = $state<string | null>(null);

	function onTableColDragStart(col: string, e: DragEvent) {
		draggedTableCol = col;
		e.dataTransfer!.effectAllowed = 'move';
		e.dataTransfer!.setData('text/plain', col);
	}

	function onTableColDragOver(col: string, e: DragEvent) {
		if (!draggedTableCol || PINNED_COLUMNS.includes(col)) return;
		e.preventDefault();
		dragOverTableCol = col;
	}

	function onTableColDrop(col: string) {
		if (draggedTableCol && draggedTableCol !== col) {
			columnOrder = moveColumn(columnOrder, draggedTableCol, col);
			logger.debug({ from: draggedTableCol, to: col }, 'Table column reordered');
		}
		draggedTableCol = null;
		dragOverTableCol = null;
	}

	function onTableColDragEnd() {
		draggedTableCol = null;
		dragOverTableCol = null;
	}

	const placeholderRows = Array.from({ length: 5 }, (_, i) => i);

	function clearFilters() {
		logger.debug('Column filters cleared');
		columnFilters = {};
	}
</script>

<section class="rounded-lg bg-surface p-6 shadow">
	<div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
		<div class="flex items-center gap-3">
			<button
				type="button"
				class="text-fg-dim hover:text-gray-700"
				onclick={() => (isCollapsed = !isCollapsed)}
				aria-label={isCollapsed ? 'Expand table' : 'Collapse table'}
			>
				<svg
					class="h-6 w-6 transition-transform duration-200"
					class:rotate-180={isCollapsed}
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
				</svg>
			</button>
			<div>
				<h2 class="text-xl font-semibold text-fg">Results</h2>
				<span class="text-sm text-fg-muted">
					{#if fetching}
						{results?.data?.length
							? `${results.data.length} rows — fetching more…`
							: 'Querying ALMA TAP…'}
					{:else if hasActiveFilters}
						{filteredData.length} of {results?.data?.length ?? 0} rows (filtered)
					{:else}
						{results?.count ? `${results.count} rows` : 'No data yet'}
					{/if}
				</span>
				{#if fetching}
					<span
						class="h-3 w-3 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"
						aria-hidden="true"
					></span>
				{/if}
			</div>
		</div>
		<div class="flex flex-wrap gap-2">
			<ResultsColumnsMenu
				bind:columnOrder
				bind:hiddenColumns
				dataColumns={dataColumnSet}
				visibleCount={tableColumns.length}
			/>
			{#if hasActiveFilters}
				<button
					type="button"
					class="rounded-md border border-amber-300 px-4 py-2 text-sm font-medium text-amber-700 hover:bg-amber-50"
					onclick={clearFilters}
				>
					Clear Filters
				</button>
			{/if}
			<button
				type="button"
				class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover disabled:cursor-not-allowed"
				onclick={onClear}
				disabled={loading || !results}
			>
				Clear Metadata
			</button>
			<button
				type="button"
				class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover"
				onclick={onLoad}
			>
				Load Metadata
			</button>
			{#if results?.data?.length}
				<button
					type="button"
					class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:bg-blue-300"
					disabled={saving}
					onclick={onSave}
				>
					{saving ? 'Saving...' : 'Save Metadata'}
				</button>
			{/if}
			{#if selectedCount > 0}
				<SelectionSizeEstimate uids={selectedMemberOusUids} count={selectedCount} />
				<button
					type="button"
					class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
					onclick={download}
				>
					Download ({selectedCount})
				</button>
			{/if}
		</div>
	</div>

	{#if !isCollapsed}
		<!-- Top mirror scrollbar: scrolls horizontally in sync with the table below -->
		<div
			bind:this={mirrorBar}
			class="mt-4 overflow-x-scroll overflow-y-hidden border-b border-line"
			style="height: 14px;"
		>
			<div style="height: 1px; width: {tableScrollWidth}px;"></div>
		</div>

		<!-- Main scroll container: vertical (capped height) + horizontal -->
		<div bind:this={scrollContainer} class="overflow-auto" style="max-height: {CONTAINER_HEIGHT}px; overscroll-behavior: contain;">
			<table class="divide-y divide-line text-sm" style="min-width: max-content; width: 100%;">
				<thead class="sticky top-0 z-10">
					<!-- Column header row -->
					<tr class="bg-surface">
						<!-- Selection checkbox header -->
						<th
							scope="col"
							class="sticky left-0 z-20 w-10 bg-surface px-2 py-2 text-center border-r border-line"
						>
							{#if results?.data?.length}
								<input
									type="checkbox"
									checked={allFilteredSelected}
									onchange={toggleSelectAll}
									class="rounded border-line"
									title="Select all"
								/>
							{/if}
						</th>
						{#each pinnedTableColumns as col, i}
							<th
								bind:this={pinnedColEls[i]}
								scope="col"
								class="sticky z-20 whitespace-nowrap bg-surface px-4 py-2 text-left font-semibold text-fg-muted border-r border-line"
								style="left: {pinnedLeft(i)}px;"
							>
								{formatColumnName(col)}
							</th>
						{/each}
						{#each reorderableTableColumns as column}
							<th
								scope="col"
								class="whitespace-nowrap px-4 py-2 text-left font-semibold text-fg-muted cursor-grab select-none"
								class:bg-accent-soft={dragOverTableCol === column && draggedTableCol !== column}
								draggable="true"
								ondragstart={(e: DragEvent) => onTableColDragStart(column, e)}
								ondragover={(e: DragEvent) => onTableColDragOver(column, e)}
								ondrop={() => onTableColDrop(column)}
								ondragend={onTableColDragEnd}
							>
								{formatColumnName(column)}
							</th>
						{/each}
					</tr>
					<!-- Filter row — only shown when there is data -->
					{#if results?.data?.length}
						<tr class="bg-surface shadow-panel">
							<!-- Empty cell for checkbox column -->
							<th scope="col" class="sticky left-0 z-20 bg-surface px-2 py-1 border-r border-line"></th>
							{#each pinnedTableColumns as column, i}
								<th scope="col" class="sticky z-20 bg-surface px-2 py-1 border-r border-line" style="left: {pinnedLeft(i)}px;">
									<input
										type="text"
										placeholder="filter (* = wildcard)"
										bind:value={columnFilters[column]}
										class="w-full min-w-16 rounded border border-line px-1.5 py-0.5 text-xs font-normal text-fg-muted placeholder-gray-300 focus:border-blue-400 focus:outline-none"
										class:border-blue-400={columnFilters[column]?.length > 0}
										class:bg-accent-soft={columnFilters[column]?.length > 0}
									/>
								</th>
							{/each}
							{#each reorderableTableColumns as column}
								<th scope="col" class="px-2 py-1">
									<input
										type="text"
										placeholder="filter (* = wildcard)"
										bind:value={columnFilters[column]}
										class="w-full min-w-16 rounded border border-line px-1.5 py-0.5 text-xs font-normal text-fg-muted placeholder-gray-300 focus:border-blue-400 focus:outline-none"
										class:border-blue-400={columnFilters[column]?.length > 0}
										class:bg-accent-soft={columnFilters[column]?.length > 0}
									/>
								</th>
							{/each}
						</tr>
					{/if}
				</thead>
				<tbody class="bg-surface">
					{#if !loading && results?.data?.length}
						<tr style="height: {rowWindow.topPad}px;" aria-hidden="true">
							<td colspan={tableColumns.length + 1}></td>
						</tr>
						{#each visibleRows as row, vi}
							{@const dataIndex = filteredIndices[rowWindow.start + vi]}
							{@const isSelected = selectedRowIndices.has(dataIndex)}
							<tr
								class="group border-b border-line hover:bg-hover"
								class:bg-accent-soft={isSelected}
								style="height: {ROW_HEIGHT}px;"
							>
								<!-- Selection checkbox -->
								<td
									class="sticky left-0 z-[5] bg-surface px-2 text-center align-middle border-r border-line group-hover:bg-gray-50"
									class:bg-accent-soft={isSelected}
								>
									<input
										type="checkbox"
										checked={isSelected}
										onchange={() => toggleRowSelection(dataIndex)}
										class="rounded border-line"
									/>
								</td>
								{#each pinnedTableColumns as column, i}
									{@const displayValue = stringifyValue(row[column])}
									<td
										class="sticky z-[5] bg-surface px-4 align-middle border-r border-line group-hover:bg-gray-50"
										style="left: {pinnedLeft(i)}px;"
									>
										<div
											class="whitespace-nowrap text-fg"
											style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;"
											title={displayValue}
										>
											{displayValue}
										</div>
									</td>
								{/each}
								{#each reorderableTableColumns as column}
									{@const displayValue = stringifyValue(row[column])}
									<td class="px-4 align-middle">
										<div
											class="whitespace-nowrap text-fg"
											style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;"
											title={displayValue}
										>
											{displayValue}
										</div>
									</td>
								{/each}
							</tr>
						{/each}
						<tr style="height: {rowWindow.bottomPad}px;" aria-hidden="true">
							<td colspan={tableColumns.length + 1}></td>
						</tr>
					{:else if !loading && hasActiveFilters}
						<tr>
							<td colspan={tableColumns.length + 1} class="px-4 py-8 text-center text-sm text-fg-dim">
								No rows match the current filters.
							</td>
						</tr>
					{:else}
						{#each placeholderRows as placeholder}
							<tr class="animate-pulse border-b border-line">
								{#each tableColumns as column}
									<td class="px-4 py-3">
										<div class="h-4 w-32 rounded bg-gray-200/80" aria-hidden={placeholder !== null && column.length >= 0}></div>
									</td>
								{/each}
							</tr>
						{/each}
					{/if}
				</tbody>
			</table>
		</div>
	{/if}
</section>
