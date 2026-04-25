<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import {
		metadataApi,
		type MetadataQuery,
		type MetadataResponse,
		type QueryPreset,
	} from '$lib/api/metadata';
	import MetadataQueryForm from '$lib/components/metadata/MetadataQueryForm.svelte';
	import MetadataResultsTable from '$lib/components/metadata/MetadataResultsTable.svelte';
	import MetadataLoadModal from '$lib/components/metadata/MetadataLoadModal.svelte';
	import MetadataSaveModal from '$lib/components/metadata/MetadataSaveModal.svelte';
	import DownloadDialog from '$lib/components/metadata/DownloadDialog.svelte';
	import FullScreenLoader from '$lib/components/shared/Loader.svelte';
	import { createLogger } from '$lib/logger';
	import { formatDateStamp, supportsFilePicker } from '$lib/utils';

	const RESULTS_CACHE_KEY = 'almasim:metadata-results';
	const DEFAULT_METADATA_PATH = 'data';
	const logger = createLogger('routes/metadata');

	type FileSystemFileHandle = {
		name?: string;
		createWritable: () => Promise<{
			write: (data: Blob) => Promise<void>;
			close: () => Promise<void>;
		}>;
	};

	type SaveFilePickerOptions = {
		suggestedName?: string;
		types?: { description?: string; accept: Record<string, string[]> }[];
	};

	// Columns required in cached data — if any are missing the cache is stale and discarded.
	const REQUIRED_CACHE_COLUMNS = ['Project_abstract', 'QA2_status', 'Type'];

	const getCachedResults = (): MetadataResponse | null => {
		if (typeof window === 'undefined') return null;
		try {
			const raw = window.localStorage.getItem(RESULTS_CACHE_KEY);
			if (!raw) return null;
			const parsed = JSON.parse(raw) as MetadataResponse;
			// Invalidate cache if it predates required columns
			const sample = parsed?.data?.[0];
			if (sample && REQUIRED_CACHE_COLUMNS.some((col) => !(col in sample))) {
				window.localStorage.removeItem(RESULTS_CACHE_KEY);
				return null;
			}
			return parsed;
		} catch {
			return null;
		}
	};

	const persistResults = (data: MetadataResponse) => {
		if (typeof window === 'undefined') return;
		try {
			window.localStorage.setItem(RESULTS_CACHE_KEY, JSON.stringify(data));
		} catch {
			/* ignore storage write failures */
		}
	};

	const snapshotMetadata = (data: MetadataResponse): MetadataResponse => ({
		count: typeof data.count === 'number' ? data.count : data.data.length,
		data: data.data.map((row) => ({ ...row }))
	});

	const rowsToCsv = (rows: Record<string, unknown>[]): string => {
		if (!rows.length) return '';
		const columnsSet = new Set<string>();
		const columns: string[] = [];
		for (const row of rows) {
			for (const key of Object.keys(row)) {
				if (!columnsSet.has(key)) {
					columnsSet.add(key);
					columns.push(key);
				}
			}
		}
		const escape = (val: unknown): string => {
			if (val === null || val === undefined) return '';
			const s = typeof val === 'string' ? val : JSON.stringify(val);
			return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
		};
		const header = columns.map(escape).join(',');
		const body = rows.map((r) => columns.map((c) => escape(r[c])).join(',')).join('\n');
		return `${header}\n${body}\n`;
	};

	const downloadMetadata = (data: MetadataResponse, path: string, format: 'json' | 'csv' = 'json') => {
		if (typeof window === 'undefined') return;
		const snapshot = snapshotMetadata(data);
		const extension = format === 'csv' ? '.csv' : '.json';
		const body =
			format === 'csv'
				? rowsToCsv(snapshot.data)
				: JSON.stringify(snapshot, null, 2);
		const mime = format === 'csv' ? 'text/csv' : 'application/json';
		const blob = new Blob([body], { type: mime });
		const safePath = path.replace(/^[.@/\\]+/, '');
		const sanitized =
			safePath.length > 0 ? safePath.replace(/\s+/g, '-').replace(/[^\w./-]/g, '_') : '';
		const segments = sanitized.split('/').filter(Boolean);
		let fileName = segments.length
			? segments.join('_')
			: `metadata-results-${formatDateStamp()}`;
		fileName = fileName.replace(/\.(json|csv)$/i, '');
		fileName += extension;
		const link = document.createElement('a');
		link.href = URL.createObjectURL(blob);
		link.download = fileName;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
		URL.revokeObjectURL(link.href);
	};

	const normalizeBackendPath = (input: string, format: 'json' | 'csv' = 'json') => {
		const extension = format === 'csv' ? '.csv' : '.json';
		const cleaned = input.replace(/^[./\\]+/, '').trim() || DEFAULT_METADATA_PATH;
		// Strip any existing json/csv extension so the format selector wins.
		const withoutExt = cleaned.replace(/\.(json|csv)$/i, '');
		// If the user provided no leaf filename (just a directory), append the default name.
		const hasFileLeaf = /[^/]+$/.test(withoutExt) && !withoutExt.endsWith('/');
		const base = hasFileLeaf
			? withoutExt
			: `${withoutExt.replace(/\/$/, '')}/metadata-results`;
		return `${base}${extension}`;
	};

	const parseCsv = (input: string): Array<Record<string, string>> => {
		const rows: string[][] = [];
		let current = '';
		let inQuotes = false;
		const currentRow: string[] = [];

		const pushValue = () => {
			currentRow.push(current);
			current = '';
		};

		const pushRow = () => {
			pushValue();
			rows.push([...currentRow]);
			currentRow.length = 0;
		};

		for (let i = 0; i < input.length; i++) {
			const char = input[i];
			if (char === '"') {
				if (inQuotes && input[i + 1] === '"') {
					current += '"';
					i++;
				} else {
					inQuotes = !inQuotes;
				}
			} else if (char === ',' && !inQuotes) {
				pushValue();
			} else if ((char === '\n' || char === '\r') && !inQuotes) {
				if (char === '\r' && input[i + 1] === '\n') i++;
				pushRow();
			} else {
				current += char;
			}
		}

		if (current.length > 0 || currentRow.length > 0) {
			pushRow();
		}

		if (!rows.length) return [];
		const [header, ...dataRows] = rows;
		return dataRows
			.filter((cells) => cells.some((cell) => cell.trim().length))
			.map((cells) =>
				header.reduce<Record<string, string>>((acc, column, index) => {
					acc[column] = cells[index] ?? '';
					return acc;
				}, {})
			);
	};

	const parseMetadataFile = async (file: File): Promise<MetadataResponse> => {
		const text = await file.text();
		const extension = file.name.split('.').pop()?.toLowerCase();
		if (extension === 'csv') {
			const data = parseCsv(text);
			return { data, count: data.length };
		}

		try {
			const parsed = JSON.parse(text) as Partial<MetadataResponse>;
			if (!parsed || !Array.isArray(parsed.data)) {
				throw new Error();
			}
			return {
				data: parsed.data,
				count:
					typeof parsed.count === 'number'
						? parsed.count
						: Array.isArray(parsed.data)
							? parsed.data.length
							: 0
			};
		} catch {
			throw new Error('Invalid metadata file. Provide JSON with a `data` array or a CSV file.');
		}
	};

	type QueryJob = {
		id: string;
		status: 'starting' | 'fetching' | 'done' | 'cancelled' | 'error';
		rows: Record<string, unknown>[];
		message: string;
		error: string;
		pollTimer: ReturnType<typeof setTimeout> | null;
		page: number;
		query: MetadataQuery;
	};

	let scienceTypes = $state<{ keywords: string[]; categories: string[] } | null>(null);
	// Loaded/cached results (from file or backend)
	let loadedResults = $state<MetadataResponse | null>(null);
	// Background query jobs
	let jobs = $state<QueryJob[]>([]);
	let starting = $state(false);
	let saving = $state(false);
	let error = $state<string>('');
	let loadModalOpen = $state(false);
	let saveModalOpen = $state(false);
	let statusMessage = $state('');
	let localSaveFileName = $state('');
	let loadFormRef = $state<HTMLFormElement>();
	let saveFormRef = $state<HTMLFormElement>();
	let localSaveHandle: FileSystemFileHandle | null = null;
	let initialLoading = $state(true);

	// Query presets
	let presets = $state<QueryPreset[]>([]);
	let presetsLoading = $state(false);
	let presetSaveOpen = $state(false);
	let presetSaveName = $state('');
	let presetSaveDesc = $state('');
	let presetSaving = $state(false);
	let presetSaveForJobId = $state<string | null>(null);

	// Download state
	let downloadDialogOpen = $state(false);
	let downloadMemberOusUids = $state<string[]>([]);
	let downloadMetadataRows = $state<Record<string, unknown>[]>([]);

	let selectedJobId = $state<string | null>(null);

	const activeJobs = $derived(jobs.filter((j) => j.status === 'fetching' || j.status === 'starting'));
	const fetching = $derived(activeJobs.length > 0);

	// When a job is selected show only its rows; otherwise merge all job rows
	const selectedJob = $derived(selectedJobId ? jobs.find((j) => j.id === selectedJobId) ?? null : null);
	const queryRows = $derived(selectedJob ? selectedJob.rows : jobs.flatMap((j) => j.rows));
	const results = $derived<MetadataResponse | null>(
		queryRows.length > 0 ? { data: queryRows, count: queryRows.length } : loadedResults
	);

	function updateJob(id: string, patch: Partial<QueryJob>) {
		jobs = jobs.map((j) => (j.id === id ? { ...j, ...patch } : j));
	}

	onDestroy(() => {
		for (const job of jobs) {
			if (job.pollTimer) clearTimeout(job.pollTimer);
		}
	});

	onMount(async () => {
		logger.info('Metadata page mounted');
		try {
			scienceTypes = await metadataApi.getScienceTypes();
			logger.info(
				{ keywords: scienceTypes.keywords.length, categories: scienceTypes.categories.length },
				'Science types loaded'
			);
		} catch (err) {
			logger.error({ err }, 'Failed to load science types');
		} finally {
			initialLoading = false;
		}

		const cached = getCachedResults();
		if (cached) {
			loadedResults = cached;
			statusMessage = `Loaded cached metadata (${cached.count ?? cached.data.length} rows)`;
			logger.debug({ rowCount: cached.count ?? cached.data.length }, 'Restored metadata from cache');
		}

		await loadPresets();
	});

	async function loadPresets() {
		presetsLoading = true;
		try {
			const resp = await metadataApi.listPresets();
			presets = resp.presets;
		} catch (err) {
			logger.error({ err }, 'Failed to load query presets');
		} finally {
			presetsLoading = false;
		}
	}

	function openPresetSave(jobId: string) {
		presetSaveForJobId = jobId;
		presetSaveName = '';
		presetSaveDesc = '';
		presetSaveOpen = true;
	}

	async function handleSavePreset() {
		if (!presetSaveName.trim()) return;
		const job = jobs.find((j) => j.id === presetSaveForJobId);
		if (!job) return;
		presetSaving = true;
		try {
			const saved = await metadataApi.savePreset({
				name: presetSaveName.trim(),
				description: presetSaveDesc.trim(),
				filters: job.query,
				result_count: job.rows.length,
			});
			presets = [saved, ...presets.filter((p) => p.name !== saved.name)];
			statusMessage = `Preset "${saved.name}" saved.`;
			presetSaveOpen = false;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to save preset';
			logger.error({ err }, 'Failed to save query preset');
		} finally {
			presetSaving = false;
		}
	}

	function applyPreset(preset: QueryPreset) {
		runQuery(preset.filters);
		statusMessage = `Running preset "${preset.name}"…`;
	}

	async function handleDeletePreset(name: string) {
		try {
			await metadataApi.deletePreset(name);
			presets = presets.filter((p) => p.name !== name);
			statusMessage = `Preset "${name}" deleted.`;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to delete preset';
			logger.error({ err, name }, 'Failed to delete query preset');
		}
	}

	async function runQuery(query: MetadataQuery) {
		starting = true;
		error = '';
		logger.info({ query }, 'Metadata query submitted');

		let queryId: string;
		try {
			const start = await metadataApi.startQuery(query);
			queryId = start.query_id;
			logger.info({ queryId }, 'Metadata query started');
		} catch (err) {
			error = err instanceof Error ? err.message : 'Unable to start metadata query';
			starting = false;
			logger.error({ err }, 'Failed to start metadata query');
			return;
		}

		const job: QueryJob = {
			id: queryId,
			status: 'fetching',
			rows: [],
			message: 'Querying ALMA TAP…',
			error: '',
			pollTimer: null,
			page: 0,
			// Use $state.snapshot to get a plain object — structuredClone trips on
			// Svelte 5 reactive proxies in some cases.
			query: $state.snapshot(query) as MetadataQuery,
		};
		jobs = [...jobs, job];
		starting = false;

		const PAGE_SIZE = 500;
		// Poll cadence — fast while rows are streaming, slow while waiting on TAP.
		const FAST_DELAY = 250;
		const IDLE_BASE_DELAY = 1500;
		const IDLE_MAX_DELAY = 8000;
		// Stop polling if TAP gives us no new rows for this long (10 minutes).
		const HANG_TIMEOUT_MS = 10 * 60 * 1000;

		let idlePolls = 0;
		let lastProgressAt = Date.now();

		async function pollNext() {
			// Read latest job state
			const current = jobs.find((j) => j.id === queryId);
			if (!current || current.status === 'cancelled') return;

			try {
				const pageData = await metadataApi.getQueryPage(queryId, current.page, PAGE_SIZE);

				if (pageData.error) {
					updateJob(queryId, { status: 'error', error: pageData.error, pollTimer: null });
					logger.error({ queryId, err: pageData.error }, 'Query job returned error');
					return;
				}

				const newRows = current.rows.concat(pageData.rows);
				const newPage = pageData.rows.length > 0 ? current.page + 1 : current.page;
				const totalKnown = pageData.total_fetched ?? newRows.length;
				const phase = pageData.status ?? 'running';

				// Reset / advance the hang detector based on row progress.
				if (pageData.rows.length > 0) {
					idlePolls = 0;
					lastProgressAt = Date.now();
				} else {
					idlePolls += 1;
				}

				let msg: string;
				if (pageData.done) {
					msg = `Done — ${newRows.length} rows`;
				} else if (phase === 'running' && totalKnown === 0) {
					msg = 'Querying ALMA TAP…';
				} else if (idlePolls > 2 && phase === 'running') {
					msg = `Waiting on ALMA TAP… ${newRows.length} of ${totalKnown} rows so far`;
				} else {
					msg = `Fetching… ${newRows.length} of ${totalKnown} rows`;
				}

				if (pageData.done) {
					updateJob(queryId, { rows: newRows, page: newPage, status: 'done', message: msg, pollTimer: null });
					if (newRows.length > 0) {
						const merged = jobs.flatMap((j) => (j.id === queryId ? newRows : j.rows));
						persistResults({ data: merged, count: merged.length });
					}
					logger.info({ queryId, totalRows: newRows.length }, 'Query job complete');
					return;
				}

				// Safety net: stop polling if we've made no progress for too long.
				if (Date.now() - lastProgressAt > HANG_TIMEOUT_MS) {
					updateJob(queryId, {
						status: 'error',
						error: 'TAP query timed out (no new rows for over 10 minutes).',
						pollTimer: null,
					});
					logger.warn(
						{ queryId, totalKnown, idlePolls },
						'Stopping poll loop after hang timeout'
					);
					return;
				}

				// Backoff: fast while rows are flowing, exponential while waiting on TAP.
				let delay: number;
				if (pageData.rows.length >= PAGE_SIZE) {
					delay = FAST_DELAY;
				} else if (pageData.rows.length > 0) {
					delay = IDLE_BASE_DELAY;
				} else {
					delay = Math.min(IDLE_BASE_DELAY * Math.pow(1.5, idlePolls - 1), IDLE_MAX_DELAY);
				}
				const timer = setTimeout(pollNext, delay);
				updateJob(queryId, { rows: newRows, page: newPage, message: msg, pollTimer: timer });
				logger.debug(
					{ queryId, page: newPage, rowsSoFar: newRows.length, phase, totalKnown, idlePolls, delay },
					'Polling next page'
				);
			} catch (err) {
				const msg = err instanceof Error ? err.message : 'Polling failed';
				updateJob(queryId, { status: 'error', error: msg, pollTimer: null });
				logger.error({ err, queryId, page: current.page }, 'Metadata polling failed');
			}
		}

		const timer = setTimeout(pollNext, 1000);
		updateJob(queryId, { pollTimer: timer });
	}

	async function stopJob(jobId: string) {
		const job = jobs.find((j) => j.id === jobId);
		if (!job) return;
		if (job.pollTimer) clearTimeout(job.pollTimer);
		updateJob(jobId, { status: 'cancelled', message: 'Cancelled', pollTimer: null });
		try {
			await metadataApi.cancelQuery(jobId);
		} catch {
			// best-effort
		}
	}

	function loadJob(jobId: string) {
		selectedJobId = selectedJobId === jobId ? null : jobId;
	}

	function saveJob(jobId: string) {
		selectedJobId = jobId;
		saveModalOpen = true;
	}

	function dismissJob(jobId: string) {
		const job = jobs.find((j) => j.id === jobId);
		if (!job) return;
		if (job.pollTimer) clearTimeout(job.pollTimer);
		if (selectedJobId === jobId) selectedJobId = null;
		jobs = jobs.filter((j) => j.id !== jobId);
	}

	async function handleLoadFile(event: Event) {
		event.preventDefault();
		const formElement = loadFormRef ?? (event.currentTarget as HTMLFormElement | undefined);
		if (!formElement) return;
		const formData = new FormData(formElement);

		const file = formData.get('file_upload');
		const filePath = (formData.get('file_path') as string)?.trim();

		if (!(file instanceof File) && !filePath) {
			error = 'Select a metadata file or provide a backend-accessible path.';
			return;
		}

		const source = file instanceof File && (file as File).size > 0 ? 'local file' : filePath;
		logger.info({ source }, 'Load metadata requested');

		starting = true;
		error = '';
		try {
			let data: MetadataResponse;
			if (file instanceof File && file.size > 0) {
				data = await parseMetadataFile(file);
			} else if (filePath) {
				data = await metadataApi.load(filePath);
			} else {
				throw new Error('Unable to determine how to load the metadata.');
			}

			loadedResults = data;
			jobs = [];  // clear query jobs so loaded data is shown
			persistResults(data);
			loadModalOpen = false;
			formElement.reset();
			statusMessage = `Loaded ${data.count} rows from ${file instanceof File ? 'local file' : filePath}`;
			logger.info({ rowCount: data.count, source }, 'Metadata loaded successfully');
		} catch (err) {
			error = err instanceof Error ? err.message : 'Unable to load metadata file';
			logger.error({ err, source }, 'Failed to load metadata file');
		} finally {
			starting = false;
		}
	}

	async function handleSaveMetadata(event: Event) {
		event.preventDefault();
		const current = results;
		if (!current || !current.data?.length) {
			error = 'No metadata to save yet. Run a query or load data first.';
			return;
		}
		const snapshot = snapshotMetadata(current);
		const formElement = saveFormRef ?? (event.currentTarget as HTMLFormElement | undefined);
		if (!formElement) return;
		const formData = new FormData(formElement);
		const rawFormat = (formData.get('save_format') as string) || 'json';
		const format: 'json' | 'csv' = rawFormat === 'csv' ? 'csv' : 'json';
		const backendPath = normalizeBackendPath(
			(formData.get('save_path') as string) || DEFAULT_METADATA_PATH,
			format
		);

		saving = true;
		error = '';
		let backendSucceeded = false;
		try {
			await metadataApi.save({ path: backendPath, data: snapshot.data, format });
			backendSucceeded = true;
			statusMessage = `Saved ${snapshot.count} rows to backend path ${backendPath}`;
			logger.info(
				{ rowCount: snapshot.count, backendPath, format },
				'Metadata saved to backend'
			);
		} catch (err) {
			const message =
				err instanceof Error
					? err.message
					: "Unable to save metadata via API. We'll keep working on a local copy.";
			error = message;
			logger.error({ err, backendPath, format }, 'Failed to save metadata to backend');
		} finally {
			saving = false;
		}

		const completeCleanup = () => {
			formElement.reset();
			localSaveHandle = null;
			localSaveFileName = '';
			saveModalOpen = false;
		};

		const needsLocalFallback = !backendSucceeded;
		try {
			if (localSaveHandle) {
				const writable = await localSaveHandle.createWritable();
				const body =
					format === 'csv'
						? rowsToCsv(snapshot.data)
						: JSON.stringify(snapshot, null, 2);
				const mime = format === 'csv' ? 'text/csv' : 'application/json';
				await writable.write(new Blob([body], { type: mime }));
				await writable.close();
				statusMessage = `Saved locally to ${localSaveFileName || localSaveHandle.name || `metadata.${format}`}${backendSucceeded ? ' (backend copy updated too)' : ''}`;
			} else if (needsLocalFallback) {
				downloadMetadata(snapshot, backendPath, format);
				statusMessage = `Downloaded metadata snapshot (${snapshot.count} rows, ${format.toUpperCase()})`;
			}
		} catch (localError) {
			logger.error({ err: localError }, 'Unable to write metadata locally');
			error = 'Unable to write metadata locally.';
		} finally {
			completeCleanup();
		}
	}

	async function chooseLocalSavePath() {
		if (!supportsFilePicker()) {
			error =
				'Your browser does not support choosing a local save location. The file will download instead.';
			return;
		}
		try {
			const picker = (
				window as Window & {
					showSaveFilePicker?: (options: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;
				}
			).showSaveFilePicker;
			if (!picker) {
				error = 'File picker API is unavailable in this environment.';
				return;
			}
			const handle = await picker({
				suggestedName: `almasim-metadata-${formatDateStamp()}.json`,
				types: [
					{
						description: 'JSON file',
						accept: { 'application/json': ['.json'] }
					}
				]
			});
			localSaveHandle = handle;
			localSaveFileName = handle.name || '';
			statusMessage = `Local save target selected: ${handle.name || 'metadata.json'}`;
		} catch (err) {
			if (err instanceof DOMException && err.name === 'AbortError') {
				return;
			}
			error = err instanceof Error ? err.message : 'Unable to open save dialog.';
		}
	}

	function clearMetadata() {
		logger.info('Metadata cleared by user');
		for (const job of jobs) {
			if (job.pollTimer) clearTimeout(job.pollTimer);
		}
		jobs = [];
		selectedJobId = null;
		loadedResults = null;
		statusMessage = 'Metadata cleared.';
		if (typeof window !== 'undefined') {
			window.localStorage.removeItem(RESULTS_CACHE_KEY);
		}
	}

	function handleDownload(memberOusUids: string[], metadataRows: Record<string, unknown>[]) {
		logger.info({ count: memberOusUids.length }, 'Download dialog opened');
		downloadMemberOusUids = memberOusUids;
		downloadMetadataRows = metadataRows;
		downloadDialogOpen = true;
	}

	function handleDownloadStarted(jobId: string) {
		logger.info({ jobId }, 'Download job started');
		statusMessage = `Download started (job ${jobId.slice(0, 8)}…). Track progress on the <a href="/data" class="underline font-medium">Data</a> page.`;
	}
</script>

<div class="container mx-auto px-4 py-8">
	{#if initialLoading}
		<FullScreenLoader />
	{/if}
	<div class="space-y-8">
		<header>
			<h1 class="text-3xl font-bold text-gray-900">Metadata Explorer</h1>
			<p class="mt-2 text-gray-600">Query ALMA observation metadata or load a precomputed dataset.</p>
		</header>

		<!-- Query Presets panel -->
	{#if presets.length > 0}
		<div class="rounded-lg border border-indigo-100 bg-indigo-50 p-4">
			<h2 class="mb-3 text-sm font-semibold text-indigo-800">Saved Query Presets</h2>
			<div class="space-y-2">
				{#each presets as preset (preset.name)}
					<div class="flex items-center justify-between rounded-md border border-indigo-200 bg-white px-3 py-2 text-sm">
						<div class="min-w-0">
							<span class="font-medium text-gray-900">{preset.name}</span>
							{#if preset.description}
								<span class="ml-2 text-xs text-gray-500">{preset.description}</span>
							{/if}
							<span class="ml-2 text-xs text-gray-400">({preset.result_count} rows)</span>
						</div>
						<div class="ml-4 flex shrink-0 gap-2">
							<button
								onclick={() => applyPreset(preset)}
								class="rounded-md bg-indigo-600 px-2 py-1 text-xs font-medium text-white hover:bg-indigo-700"
							>Apply & Run</button>
							<button
								onclick={() => handleDeletePreset(preset.name)}
								class="rounded-md bg-white px-2 py-1 text-xs font-medium text-red-600 ring-1 ring-red-200 hover:bg-red-50"
							>Delete</button>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<MetadataQueryForm {scienceTypes} loading={starting} onSubmit={runQuery} />

		{#if error}
			<div class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
				{error}
			</div>
		{/if}

		{#if statusMessage}
			<div class="rounded-md border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
				{@html statusMessage}
			</div>
		{/if}

		<!-- Per-job status panel -->
		{#if jobs.length > 0}
			<div class="space-y-2">
				{#if selectedJobId}
					<div class="flex items-center justify-between rounded-md border border-indigo-200 bg-indigo-50 px-4 py-2 text-xs text-indigo-700">
						<span>Showing results for query <span class="font-mono">{selectedJobId.slice(0, 8)}</span></span>
						<button onclick={() => (selectedJobId = null)} class="underline hover:text-indigo-900">Show all</button>
					</div>
				{/if}
				{#each jobs as job (job.id)}
					<div class="flex items-center justify-between rounded-md border px-4 py-2 text-sm
						{selectedJobId === job.id ? 'ring-2 ring-indigo-400' : ''}
						{job.status === 'error' ? 'border-red-200 bg-red-50 text-red-800' :
						 job.status === 'cancelled' ? 'border-gray-200 bg-gray-50 text-gray-500' :
						 job.status === 'done' ? 'border-green-200 bg-green-50 text-green-800' :
						 'border-blue-200 bg-blue-50 text-blue-800'}">
						<div class="flex items-center gap-2 min-w-0">
							{#if job.status === 'fetching' || job.status === 'starting'}
								<svg class="h-4 w-4 shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
									<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
									<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
								</svg>
							{/if}
							<span class="truncate">
								{job.status === 'error' ? job.error : job.message}
								<span class="ml-1 font-mono text-xs opacity-60">{job.id.slice(0, 8)}</span>
							</span>
						</div>
						<div class="ml-4 flex shrink-0 gap-2">
							{#if job.status === 'fetching'}
								<button
									onclick={() => stopJob(job.id)}
									class="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700"
								>Stop</button>
							{:else}
								{#if job.status === 'done' && job.rows.length > 0}
									<button
										onclick={() => loadJob(job.id)}
										class="rounded-md px-2 py-1 text-xs font-medium ring-1
											{selectedJobId === job.id
												? 'bg-indigo-600 text-white ring-indigo-600 hover:bg-indigo-700'
												: 'bg-white text-indigo-700 ring-indigo-300 hover:bg-indigo-50'}"
									>{selectedJobId === job.id ? 'Loaded' : 'Load'}</button>
									<button
										onclick={() => saveJob(job.id)}
										class="rounded-md bg-white px-2 py-1 text-xs font-medium text-gray-700 ring-1 ring-gray-300 hover:bg-gray-50"
									>Save</button>
									<button
										onclick={() => openPresetSave(job.id)}
										class="rounded-md bg-white px-2 py-1 text-xs font-medium text-indigo-700 ring-1 ring-indigo-300 hover:bg-indigo-50"
									>Save as Preset</button>
								{/if}
								<button
									onclick={() => dismissJob(job.id)}
									class="rounded-md bg-white px-2 py-1 text-xs font-medium text-gray-600 ring-1 ring-gray-300 hover:bg-gray-50"
								>Dismiss</button>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<MetadataResultsTable
			{results}
			loading={starting}
			{fetching}
			onClear={clearMetadata}
			onLoad={() => (loadModalOpen = true)}
			onSave={() => (saveModalOpen = true)}
			{saving}
			onDownload={handleDownload}
		/>

		<MetadataLoadModal
			open={loadModalOpen}
			loading={starting}
			defaultPath={DEFAULT_METADATA_PATH}
			onClose={() => (loadModalOpen = false)}
			onSubmit={handleLoadFile}
			bind:formRef={loadFormRef}
			{presets}
			{presetsLoading}
			onApplyPreset={(preset) => {
				loadModalOpen = false;
				applyPreset(preset);
			}}
		/>

		<MetadataSaveModal
			open={saveModalOpen}
			{saving}
			defaultPath={DEFAULT_METADATA_PATH}
			localFileName={localSaveFileName}
			onClose={() => (saveModalOpen = false)}
			onSubmit={handleSaveMetadata}
			onChooseLocalPath={chooseLocalSavePath}
			bind:formRef={saveFormRef}
		/>

		<DownloadDialog
			open={downloadDialogOpen}
			memberOusUids={downloadMemberOusUids}
			metadataRows={downloadMetadataRows}
			onClose={() => (downloadDialogOpen = false)}
			onStarted={handleDownloadStarted}
		/>
	</div>
</div>

<!-- Preset save modal -->
{#if presetSaveOpen}
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
		<div class="w-full max-w-sm rounded-lg bg-white p-6 shadow-xl">
			<h3 class="mb-4 text-base font-semibold text-gray-900">Save Query as Preset</h3>
			<div class="space-y-3">
				<label class="block">
					<span class="text-sm font-medium text-gray-700">Preset name <span class="text-red-500">*</span></span>
					<input
						type="text"
						bind:value={presetSaveName}
						placeholder="e.g. galaxies_band6"
						class="mt-1 w-full rounded-md border border-gray-300 p-2 text-sm"
					/>
				</label>
				<label class="block">
					<span class="text-sm font-medium text-gray-700">Description</span>
					<input
						type="text"
						bind:value={presetSaveDesc}
						placeholder="Optional description"
						class="mt-1 w-full rounded-md border border-gray-300 p-2 text-sm"
					/>
				</label>
			</div>
			<div class="mt-5 flex justify-end gap-2">
				<button
					onclick={() => (presetSaveOpen = false)}
					class="rounded-md px-3 py-2 text-sm font-medium text-gray-600 ring-1 ring-gray-300 hover:bg-gray-50"
				>Cancel</button>
				<button
					onclick={handleSavePreset}
					disabled={presetSaving || !presetSaveName.trim()}
					class="rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:bg-gray-400"
				>{presetSaving ? 'Saving…' : 'Save'}</button>
			</div>
		</div>
	</div>
{/if}
