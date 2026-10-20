<script lang="ts">
	import { formatBytes } from '#lib/utils/format.js';
	import type { MetadataResponse } from '#lib/api/metadata.js';
	import { downloadApi } from '#lib/api/download.js';
	import { createLogger } from '#lib/logger.js';

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
		onDownload: (memberOusUids: string[], metadataRows: Record<string, unknown>[]) => void;
	}

	let { results, resultsKey, loading, fetching, onClear, onLoad, onSave, saving, onDownload }: Props =
		$props();

	let isCollapsed = $state(false);
	let columnsMenuOpen = $state(false);

	// Pinned columns — always first, always visible, not reorderable
	const pinnedColumns = ['proposal_id', 'ALMA_source_name'];

	// All known columns with their display labels (kept in preferred display order)
	const columnLabels: Record<string, string> = {
		ALMA_source_name: 'Source Name',
		Band: 'Band',
		Array_type: 'Array Type',
		antenna_arrays: 'Antenna Arrays',
		'Ang.res.': 'Ang. Res. (arcsec)',
		'Obs.date': 'Obs. Date',
		Project_abstract: 'Project Abstract',
		science_keyword: 'Science Keyword',
		scientific_category: 'Science Category',
		QA2_status: 'QA2 Status',
		Type: 'Project Type',
		PWV: 'PWV',
		SB_name: 'SB Name',
		'Vel.res.': 'Vel. Res. (km/s)',
		RA: 'R.A. (deg)',
		Dec: 'Dec (deg)',
		FOV: 'FOV (arcsec)',
		'Int.Time': 'Int. Time (s)',
		Cont_sens_mJybeam: 'Cont. Sens. (mJy/beam)',
		Line_sens_10kms_mJybeam: 'Line Sens. 10km/s (mJy/beam)',
		Bandwidth: 'Bandwidth (GHz)',
		Freq: 'Frequency (GHz)',
		'Freq.sup.': 'Freq. Support',
		proposal_id: 'Proposal ID',
		member_ous_uid: 'Member OUS UID',
		group_ous_uid: 'Group OUS UID',
	};

	// Preferred display order (first columns shown in table)
	const preferredOrder = [
		'Band',
		'Array_type',
		'Ang.res.',
		'Obs.date',
		'Project_abstract',
		'science_keyword',
		'scientific_category',
		'QA2_status',
		'Type',
	];

	// Default order for reorderable columns (excludes pinned)
	const defaultColumnOrder = [
		...preferredOrder,
		...Object.keys(columnLabels).filter(
			(c) => !preferredOrder.includes(c) && !pinnedColumns.includes(c)
		),
	];

	// User-reorderable column order (mutable)
	let columnOrder = $state([...defaultColumnOrder]);

	// Columns hidden by default (verbose / less commonly needed)
	const defaultHidden = new Set([
		'antenna_arrays',
		'PWV',
		'SB_name',
		'Vel.res.',
		'Cont_sens_mJybeam',
		'Line_sens_10kms_mJybeam',
		'Freq.sup.',
		'group_ous_uid',
	]);
	let hiddenColumns = $state(new Set<string>(defaultHidden));

	// Client-side column filters (reset on new results)
	let columnFilters = $state<Record<string, string>>({});


	const hasActiveFilters = $derived(Object.values(columnFilters).some((v) => v.length > 0));

	// Columns that actually appear in the current result data
	const dataColumnSet = $derived.by(() => {
		const data = results?.data;
		if (!data || data.length === 0) return new Set<string>();
		return new Set(Object.keys(data[0]));
	});

	// Visible pinned columns (present in data)
	const pinnedTableColumns = $derived(pinnedColumns.filter((c) => dataColumnSet.has(c)));

	// Visible reorderable columns
	const reorderableTableColumns = $derived(
		columnOrder.filter((c) => !hiddenColumns.has(c) && dataColumnSet.has(c))
	);

	// Full ordered list of visible columns (pinned first, then reorderable)
	const tableColumns = $derived([...pinnedTableColumns, ...reorderableTableColumns]);

	// Total column count for display
	const totalColumnCount = $derived(pinnedColumns.length + columnOrder.length);

	// Apply column filters to data
	// Supports glob-style wildcards: * matches any sequence of characters.
	// e.g. "2023*.L" matches anything starting with "2023" and ending with ".L"
	// Without *, plain substring matching is used.
	function matchesFilter(cellValue: string, pattern: string): boolean {
		const cell = cellValue.toLowerCase();
		const pat = pattern.toLowerCase();
		if (!pat.includes('*')) return cell.includes(pat);
		// Convert glob to regex: escape regex chars, replace * with .*
		const escaped = pat.replace(/([.+?^${}()|[\]\\])/g, '\\$1').replace(/\*/g, '.*');
		try {
			return new RegExp(`^${escaped}$`).test(cell);
		} catch {
			return cell.includes(pat);
		}
	}

	// Positions in results.data of the rows that pass the column filters.
	const filteredIndices = $derived.by(() => {
		const data = results?.data ?? [];
		const active = Object.entries(columnFilters).filter(([, v]) => v.length > 0);
		const indices: number[] = [];
		data.forEach((row, index) => {
			if (active.every(([col, val]) => matchesFilter(String(row[col] ?? ''), val))) {
				indices.push(index);
			}
		});
		return indices;
	});
	const filteredData = $derived(filteredIndices.map((index) => (results?.data ?? [])[index]));

	// --- Row selection (must come after filteredData) ---
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

	function getSelectedMemberOusUids(): string[] {
		const data = results?.data ?? [];
		const uids: string[] = [];
		for (const idx of selectedRowIndices) {
			const row = data[idx];
			if (row?.member_ous_uid) uids.push(String(row.member_ous_uid));
		}
		const unique = [...new Set(uids)];
		logger.info({ selectedRows: selectedRowIndices.size, uniqueUids: unique.length }, 'Download initiated from table');
		return unique;
	}

	function getSelectedMetadataRows(): Record<string, unknown>[] {
		return [...selectedRowIndices]
			.map((idx) => results?.data?.[idx])
			.filter((row): row is Record<string, unknown> => Boolean(row))
			.map((row) => ({ ...row }));
	}

	// --- Compute total size of selected observations ---
	let sizeResolving = $state(false);
	let sizeResolved = $state(0);    // UIDs resolved so far
	let sizeTotal = $state(0);       // total UIDs to resolve
	let sizeTotalBytes = $state(0);  // accumulated bytes
	let sizeTotalFiles = $state(0);  // accumulated file count
	let sizeError = $state<string>('');
	let sizeCancelled = $state(false);

	// Reset size result when selection changes
	$effect(() => {
		if (selectedRowIndices.size >= 0) {
		sizeResolved = 0;
		sizeTotal = 0;
		sizeTotalBytes = 0;
		sizeTotalFiles = 0;
		sizeError = '';
		sizeCancelled = false;
		}
	});

	function cancelSizeComputation() {
		sizeCancelled = true;
	}

	async function computeSelectedSize() {
		const uids = getSelectedMemberOusUids();
		if (uids.length === 0) return;
		sizeResolving = true;
		sizeError = '';
		sizeResolved = 0;
		sizeTotal = uids.length;
		sizeTotalBytes = 0;
		sizeTotalFiles = 0;
		sizeCancelled = false;

		try {
			for (let i = 0; i < uids.length; i++) {
				if (sizeCancelled) {
					logger.info({ resolved: i, total: uids.length }, 'Size computation cancelled');
					break;
				}
				const result = await downloadApi.resolveProducts([uids[i]]);
				sizeTotalBytes += result.products.reduce(
					(total, product) => total + (product.content_length || 0),
					0
				);
				sizeTotalFiles += result.products.length;
				sizeResolved = i + 1;
			}
			if (!sizeCancelled) {
				logger.info({ totalSize: formatBytes(sizeTotalBytes), files: sizeTotalFiles }, 'Size computed');
			}
		} catch (e) {
			sizeError = e instanceof Error ? e.message : 'Failed to compute size';
			logger.error({ err: e }, 'Failed to compute size');
		} finally {
			sizeResolving = false;
		}
	}

	const sizeFinished = $derived(!sizeResolving && sizeResolved > 0);


	// Virtual scrolling constants
	const ROW_HEIGHT = 36;        // px — matches row height below
	const OVERSCAN = 15;          // extra rows above/below viewport
	const CONTAINER_HEIGHT = 480; // must match max-height in CSS

	// Mirror-scroll: top scrollbar synced to the real scroll container
	let scrollContainer = $state<HTMLDivElement | null>(null);
	let mirrorBar = $state<HTMLDivElement | null>(null);
	let tableScrollWidth = $state(0);
	let scrollTop = $state(0);

	// Virtual window derived from scroll position
	const totalRows = $derived(filteredData.length);
	const virtualStart = $derived(Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN));
	const virtualEnd = $derived(
		Math.min(totalRows, Math.ceil((scrollTop + CONTAINER_HEIGHT) / ROW_HEIGHT) + OVERSCAN)
	);
	const topPad = $derived(virtualStart * ROW_HEIGHT);
	const bottomPad = $derived(Math.max(0, (totalRows - virtualEnd) * ROW_HEIGHT));
	const visibleRows = $derived(filteredData.slice(virtualStart, virtualEnd));

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
		if (!draggedTableCol || pinnedColumns.includes(col)) return;
		e.preventDefault();
		dragOverTableCol = col;
	}

	function onTableColDrop(col: string) {
		if (!draggedTableCol || draggedTableCol === col) {
			draggedTableCol = null;
			dragOverTableCol = null;
			return;
		}
		const order = [...columnOrder];
		const fromIdx = order.indexOf(draggedTableCol);
		const toIdx = order.indexOf(col);
		if (fromIdx !== -1 && toIdx !== -1) {
			order.splice(fromIdx, 1);
			order.splice(toIdx, 0, draggedTableCol);
			columnOrder = order;
			logger.debug({ from: draggedTableCol, to: col }, 'Table column reordered');
		}
		draggedTableCol = null;
		dragOverTableCol = null;
	}

	function onTableColDragEnd() {
		draggedTableCol = null;
		dragOverTableCol = null;
	}

	// --- Drag-and-drop: column selector menu ---
	let draggedMenuCol = $state<string | null>(null);
	let dragOverMenuCol = $state<string | null>(null);

	function onMenuDragStart(col: string, e: DragEvent) {
		draggedMenuCol = col;
		e.dataTransfer!.effectAllowed = 'move';
		e.dataTransfer!.setData('text/plain', col);
	}

	function onMenuDragOver(col: string, e: DragEvent) {
		if (!draggedMenuCol) return;
		e.preventDefault();
		dragOverMenuCol = col;
	}

	function onMenuDrop(col: string) {
		if (!draggedMenuCol || draggedMenuCol === col) {
			draggedMenuCol = null;
			dragOverMenuCol = null;
			return;
		}
		const order = [...columnOrder];
		const fromIdx = order.indexOf(draggedMenuCol);
		const toIdx = order.indexOf(col);
		if (fromIdx !== -1 && toIdx !== -1) {
			order.splice(fromIdx, 1);
			order.splice(toIdx, 0, draggedMenuCol);
			columnOrder = order;
		}
		draggedMenuCol = null;
		dragOverMenuCol = null;
	}

	function onMenuDragEnd() {
		draggedMenuCol = null;
		dragOverMenuCol = null;
	}

	const placeholderRows = Array.from({ length: 5 }, (_, i) => i);

	function formatColumnName(column: string) {
		if (columnLabels[column]) return columnLabels[column];
		return column
			.split('_')
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	}

	function stringifyValue(value: unknown) {
		if (Array.isArray(value)) return value.join(', ');
		if (typeof value === 'object' && value) return JSON.stringify(value);
		return value?.toString() ?? '';
	}

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

	function clearFilters() {
		logger.debug('Column filters cleared');
		columnFilters = {};
	}
</script>

<svelte:window
	onclick={(e) => {
		if (!(e.target as HTMLElement).closest('[data-columns-menu]')) columnsMenuOpen = false;
	}}
/>

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
			<!-- Column visibility toggle -->
			<div class="relative" data-columns-menu>
				<button
					type="button"
					class="rounded-md border border-line px-4 py-2 text-sm font-medium text-fg-muted hover:bg-hover"
					onclick={() => (columnsMenuOpen = !columnsMenuOpen)}
				>
					Columns ({tableColumns.length}/{totalColumnCount})
				</button>
				{#if columnsMenuOpen}
					<div
						class="absolute right-0 z-30 mt-1 max-h-96 w-64 overflow-y-auto rounded-md border border-line bg-surface shadow-panel"
					>
						<div class="flex items-center justify-between border-b border-line px-3 py-2">
							<span class="text-xs font-semibold uppercase tracking-wide text-fg-dim"
								>Show / Hide</span
							>
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
						{#each pinnedColumns as col}
							<div
								class="flex items-center gap-2 bg-surface px-3 py-1.5 text-sm text-fg-dim"
							>
								<svg class="h-3.5 w-3.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
									<path fill-rule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clip-rule="evenodd" />
								</svg>
								{formatColumnName(col)}
								<span class="ml-auto text-xs text-gray-300">pinned</span>
							</div>
						{/each}
						<!-- Reorderable columns (drag up/down to reorder) -->
						{#each columnOrder as col}
							<label
								class="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm text-fg-muted hover:bg-hover transition-colors"
								class:text-fg-dim={!dataColumnSet.has(col)}
								class:border-t-2={dragOverMenuCol === col && draggedMenuCol !== col}
								class:border-blue-400={dragOverMenuCol === col && draggedMenuCol !== col}
								draggable="true"
								ondragstart={(e: DragEvent) => onMenuDragStart(col, e)}
								ondragover={(e: DragEvent) => onMenuDragOver(col, e)}
								ondrop={() => onMenuDrop(col)}
								ondragend={onMenuDragEnd}
								role="listitem"
							>
								<span class="cursor-grab text-gray-300 hover:text-gray-500 select-none" aria-label="Drag to reorder">⠇⠇</span>
								<input
									type="checkbox"
									checked={!hiddenColumns.has(col)}
									onchange={() => toggleColumn(col)}
									class="rounded border-line"
								/>
								{formatColumnName(col)}
								{#if !dataColumnSet.has(col)}
									<span class="ml-auto text-xs text-gray-300">—</span>
								{/if}
							</label>
						{/each}
					</div>
				{/if}
			</div>
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
				{#if sizeResolving}
					<span class="inline-flex items-center gap-2 rounded-md border border-accent-line bg-accent-soft px-3 py-2 text-sm font-medium text-accent">
						<svg class="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
							<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25" />
							<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
						</svg>
						{sizeResolved}/{sizeTotal}
						{#if sizeTotalBytes > 0}
							— {formatBytes(sizeTotalBytes)}
						{/if}
					</span>
					<button
						type="button"
						class="rounded-md border border-danger-line px-3 py-2 text-sm font-medium text-danger hover:bg-danger-soft"
						onclick={cancelSizeComputation}
						title="Stop computing size"
					>
						Stop
					</button>
				{:else}
					<button
						type="button"
						class="rounded-md border border-accent-line px-4 py-2 text-sm font-medium text-accent hover:bg-accent-soft"
						onclick={computeSelectedSize}
						title="Compute total download size for selected observations"
					>
						Compute Size ({selectedCount})
					</button>
				{/if}
				{#if sizeFinished}
					<span class="inline-flex items-center gap-1.5 rounded-md border border-accent-line bg-accent-soft px-3 py-2 text-sm font-medium text-accent">
						{formatBytes(sizeTotalBytes)} — {sizeTotalFiles} files
						{#if sizeCancelled}
							<span class="text-xs text-amber-600">({sizeResolved}/{sizeTotal} resolved)</span>
						{/if}
					</span>
				{:else if sizeError}
					<span class="inline-flex items-center gap-1.5 rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger">
						{sizeError}
						{#if sizeResolved > 0}
							({sizeResolved}/{sizeTotal} resolved — {formatBytes(sizeTotalBytes)})
						{/if}
					</span>
				{/if}
				<button
					type="button"
					class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
					onclick={() => onDownload(getSelectedMemberOusUids(), getSelectedMetadataRows())}
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
						<tr style="height: {topPad}px;" aria-hidden="true">
							<td colspan={tableColumns.length + 1}></td>
						</tr>
						{#each visibleRows as row, vi}
							{@const dataIndex = filteredIndices[virtualStart + vi]}
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
						<tr style="height: {bottomPad}px;" aria-hidden="true">
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
