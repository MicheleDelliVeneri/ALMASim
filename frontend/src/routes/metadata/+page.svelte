<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import {
		metadataApi,
		type MetadataQuery,
		type MetadataResponse,
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

	const downloadMetadata = (data: MetadataResponse, path: string) => {
		if (typeof window === 'undefined') return;
		const snapshot = snapshotMetadata(data);
		const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
			type: 'application/json'
		});
		const safePath = path.replace(/^[.@/\\]+/, '');
		const sanitized =
			safePath.length > 0 ? safePath.replace(/\s+/g, '-').replace(/[^\w./-]/g, '_') : '';
		const segments = sanitized.split('/').filter(Boolean);
		let fileName = segments.length
			? segments.join('_')
			: `metadata-results-${formatDateStamp()}`;
		if (!fileName.endsWith('.json')) fileName += '.json';
		const link = document.createElement('a');
		link.href = URL.createObjectURL(blob);
		link.download = fileName;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
		URL.revokeObjectURL(link.href);
	};

	const normalizeBackendPath = (input: string) => {
		const cleaned = input.replace(/^[./\\]+/, '').trim() || DEFAULT_METADATA_PATH;
		if (cleaned.endsWith('.json')) return cleaned;
		return `${cleaned.replace(/\/$/, '')}/metadata-results.json`;
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

	// Download state
	let downloadDialogOpen = $state(false);
	let downloadMemberOusUids = $state<string[]>([]);

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
	});

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
		};
		jobs = [...jobs, job];
		starting = false;

		const PAGE_SIZE = 500;

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
				const msg = pageData.done
					? `Done — ${newRows.length} rows`
					: `Fetching… ${newRows.length} rows`;

				if (pageData.done) {
					updateJob(queryId, { rows: newRows, page: newPage, status: 'done', message: msg, pollTimer: null });
					if (newRows.length > 0) {
						const merged = jobs.flatMap((j) => (j.id === queryId ? newRows : j.rows));
						persistResults({ data: merged, count: merged.length });
					}
					logger.info({ queryId, totalRows: newRows.length }, 'Query job complete');
					return;
				}

				const delay = pageData.rows.length >= PAGE_SIZE ? 200 : 1500;
				const timer = setTimeout(pollNext, delay);
				updateJob(queryId, { rows: newRows, page: newPage, message: msg, pollTimer: timer });
				logger.debug({ queryId, page: newPage, rowsSoFar: newRows.length }, 'Polling next page');
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
		const backendPath = normalizeBackendPath(
			(formData.get('save_path') as string) || DEFAULT_METADATA_PATH
		);

		saving = true;
		error = '';
		let backendSucceeded = false;
		try {
			await metadataApi.save({ path: backendPath, data: snapshot.data });
			backendSucceeded = true;
			statusMessage = `Saved ${snapshot.count} rows to backend path ${backendPath}`;
			logger.info({ rowCount: snapshot.count, backendPath }, 'Metadata saved to backend');
		} catch (err) {
			const message =
				err instanceof Error
					? err.message
					: "Unable to save metadata via API. We'll keep working on a local copy.";
			error = message;
			logger.error({ err, backendPath }, 'Failed to save metadata to backend');
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
				await writable.write(
					new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' })
				);
				await writable.close();
				statusMessage = `Saved locally to ${localSaveFileName || localSaveHandle.name || 'metadata.json'}${backendSucceeded ? ' (backend copy updated too)' : ''}`;
			} else if (needsLocalFallback) {
				downloadMetadata(snapshot, backendPath);
				statusMessage = `Downloaded metadata snapshot (${snapshot.count} rows)`;
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

	function handleDownload(memberOusUids: string[]) {
		logger.info({ count: memberOusUids.length }, 'Download dialog opened');
		downloadMemberOusUids = memberOusUids;
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
			onClose={() => (downloadDialogOpen = false)}
			onStarted={handleDownloadStarted}
		/>
	</div>
</div>
