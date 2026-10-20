<script lang="ts">
	import {
		clearCachedResults,
		readCachedResults,
		writeCachedResults
	} from '#lib/utils/metadataCache.js';
	import { resolve } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';
	import {
		metadataApi,
		type MetadataQuery,
		type MetadataResponse,
		type MetadataRow,
		type QueryPreset
	} from '#lib/api/metadata.js';
	import MetadataQueryForm from '#lib/components/metadata/MetadataQueryForm.svelte';
	import MetadataResultsTable from '#lib/components/metadata/MetadataResultsTable.svelte';
	import MetadataLoadModal from '#lib/components/metadata/MetadataLoadModal.svelte';
	import MetadataSaveModal from '#lib/components/metadata/MetadataSaveModal.svelte';
	import DownloadDialog from '#lib/components/metadata/DownloadDialog.svelte';
	import FullScreenLoader from '#lib/components/shared/Loader.svelte';
	import PresetSaveModal from '#lib/components/metadata/PresetSaveModal.svelte';
	import QueryJobsPanel from '#lib/components/metadata/QueryJobsPanel.svelte';
	import QueryPresetsPanel from '#lib/components/metadata/QueryPresetsPanel.svelte';
	import type { QueryJob } from '#lib/components/metadata/queryJob.js';
	import { createLogger } from '#lib/logger.js';
	import { formatDateStamp, supportsFilePicker } from '#lib/utils.js';
	import {
		DEFAULT_METADATA_PATH,
		downloadMetadata,
		normalizeBackendPath,
		parseMetadataFile,
		rowsToCsv,
		snapshotMetadata
	} from '#lib/utils/metadataFiles.js';

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
	let statusLink = $state<{ route: '/data'; label: string } | null>(null);

	/** Show a status line, optionally followed by a link. */
	function setStatus(message: string, link: { route: '/data'; label: string } | null = null) {
		statusMessage = message;
		statusLink = link;
	}
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
	let downloadMetadataRows = $state<MetadataRow[]>([]);

	let selectedJobId = $state<string | null>(null);

	const activeJobs = $derived(
		jobs.filter((j) => j.status === 'fetching' || j.status === 'starting')
	);
	const fetching = $derived(activeJobs.length > 0);

	// When a job is selected show only its rows; otherwise merge all job rows
	const selectedJob = $derived(
		selectedJobId ? (jobs.find((j) => j.id === selectedJobId) ?? null) : null
	);
	const queryRows = $derived(selectedJob ? selectedJob.rows : jobs.flatMap((j) => j.rows));
	const results = $derived<MetadataResponse | null>(
		queryRows.length > 0 ? { data: queryRows, count: queryRows.length } : loadedResults
	);
	// What the table shows, stable while a query streams more rows into it.
	const resultsKey = $derived<unknown>(
		queryRows.length > 0
			? `jobs:${selectedJobId ?? jobs.map((j) => j.id).join(',')}`
			: loadedResults
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

		const cached = readCachedResults();
		if (cached) {
			loadedResults = cached;
			setStatus(`Loaded cached metadata (${cached.count ?? cached.data.length} rows)`);
			logger.debug(
				{ rowCount: cached.count ?? cached.data.length },
				'Restored metadata from cache'
			);
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
				result_count: job.rows.length
			});
			presets = [saved, ...presets.filter((p) => p.name !== saved.name)];
			setStatus(`Preset "${saved.name}" saved.`);
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
		setStatus(`Running preset "${preset.name}"…`);
	}

	async function handleDeletePreset(name: string) {
		try {
			await metadataApi.deletePreset(name);
			presets = presets.filter((p) => p.name !== name);
			setStatus(`Preset "${name}" deleted.`);
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
			query: $state.snapshot(query) as MetadataQuery
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
					updateJob(queryId, {
						rows: newRows,
						page: newPage,
						status: 'done',
						message: msg,
						pollTimer: null
					});
					if (newRows.length > 0) {
						const merged = jobs.flatMap((j) => (j.id === queryId ? newRows : j.rows));
						writeCachedResults({ data: merged, count: merged.length });
					}
					logger.info({ queryId, totalRows: newRows.length }, 'Query job complete');
					return;
				}

				// Safety net: stop polling if we've made no progress for too long.
				if (Date.now() - lastProgressAt > HANG_TIMEOUT_MS) {
					updateJob(queryId, {
						status: 'error',
						error: 'TAP query timed out (no new rows for over 10 minutes).',
						pollTimer: null
					});
					logger.warn({ queryId, totalKnown, idlePolls }, 'Stopping poll loop after hang timeout');
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
					{
						queryId,
						page: newPage,
						rowsSoFar: newRows.length,
						phase,
						totalKnown,
						idlePolls,
						delay
					},
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
			jobs = []; // clear query jobs so loaded data is shown
			writeCachedResults(data);
			loadModalOpen = false;
			formElement.reset();
			setStatus(`Loaded ${data.count} rows from ${file instanceof File ? 'local file' : filePath}`);
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
			setStatus(`Saved ${snapshot.count} rows to backend path ${backendPath}`);
			logger.info({ rowCount: snapshot.count, backendPath, format }, 'Metadata saved to backend');
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
					format === 'csv' ? rowsToCsv(snapshot.data) : JSON.stringify(snapshot, null, 2);
				const mime = format === 'csv' ? 'text/csv' : 'application/json';
				await writable.write(new Blob([body], { type: mime }));
				await writable.close();
				setStatus(
					`Saved locally to ${localSaveFileName || localSaveHandle.name || `metadata.${format}`}${backendSucceeded ? ' (backend copy updated too)' : ''}`
				);
			} else if (needsLocalFallback) {
				downloadMetadata(snapshot, backendPath, format);
				setStatus(`Downloaded metadata snapshot (${snapshot.count} rows, ${format.toUpperCase()})`);
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
			setStatus(`Local save target selected: ${handle.name || 'metadata.json'}`);
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
		setStatus('Metadata cleared.');
		clearCachedResults();
	}

	function handleDownload(memberOusUids: string[], metadataRows: MetadataRow[]) {
		logger.info({ count: memberOusUids.length }, 'Download dialog opened');
		downloadMemberOusUids = memberOusUids;
		downloadMetadataRows = metadataRows;
		downloadDialogOpen = true;
	}

	function handleDownloadStarted(jobId: string) {
		logger.info({ jobId }, 'Download job started');
		setStatus(`Download started (job ${jobId.slice(0, 8)}…). Track progress on the`, {
			route: '/data',
			label: 'Data page'
		});
	}
</script>

<div class="container mx-auto px-4 py-8">
	{#if initialLoading}
		<FullScreenLoader />
	{/if}
	<div class="space-y-8">
		<header>
			<h1 class="text-3xl font-bold text-fg">Metadata Explorer</h1>
			<p class="mt-2 text-fg-muted">
				Query ALMA observation metadata or load a precomputed dataset.
			</p>
		</header>

		<QueryPresetsPanel {presets} onApply={applyPreset} onDelete={handleDeletePreset} />

		<MetadataQueryForm {scienceTypes} loading={starting} onSubmit={runQuery} />

		{#if error}
			<div class="rounded-md border border-danger-line bg-danger-soft p-4 text-sm text-danger">
				{error}
			</div>
		{/if}

		{#if statusMessage}
			<div
				class="whitespace-pre-wrap rounded-md border border-accent-line bg-accent-soft p-4 text-sm text-accent"
			>
				{statusMessage}
				{#if statusLink}
					<a href={resolve(statusLink.route)} class="font-medium underline">{statusLink.label}</a>.
				{/if}
			</div>
		{/if}

		<QueryJobsPanel
			{jobs}
			{selectedJobId}
			onShowAll={() => (selectedJobId = null)}
			onStop={stopJob}
			onLoad={loadJob}
			onSave={saveJob}
			onSavePreset={openPresetSave}
			onDismiss={dismissJob}
		/>

		<MetadataResultsTable
			{results}
			{resultsKey}
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

<PresetSaveModal
	open={presetSaveOpen}
	saving={presetSaving}
	bind:name={presetSaveName}
	bind:description={presetSaveDesc}
	onCancel={() => (presetSaveOpen = false)}
	onSave={handleSavePreset}
/>
