<script lang="ts">
	import { onMount } from 'svelte';
	import {
		metadataApi,
		type MetadataQuery,
		type MetadataResponse,
		type MetadataSaveRequest
	} from '$lib/api/metadata';
	import MetadataQueryForm from '$lib/components/metadata/MetadataQueryForm.svelte';
	import MetadataResultsTable from '$lib/components/metadata/MetadataResultsTable.svelte';
	import MetadataLoadModal from '$lib/components/metadata/MetadataLoadModal.svelte';
	import MetadataSaveModal from '$lib/components/metadata/MetadataSaveModal.svelte';
	import FullScreenLoader from '$lib/components/shared/Loader.svelte';
	import { formatDateStamp, supportsFilePicker } from '$lib/utils';

	const RESULTS_CACHE_KEY = 'almasim:metadata-results';
	const DEFAULT_METADATA_PATH = 'data';

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

	const downloadMetadata = (data: MetadataResponse, path: string) => {
		if (typeof window === 'undefined') return;
		const blob = new Blob([JSON.stringify(data, null, 2)], {
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

	let scienceTypes = $state<{ keywords: string[]; categories: string[] } | null>(null);
	let results = $state<MetadataResponse | null>(null);
	let loading = $state(false);
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

	onMount(async () => {
		try {
			scienceTypes = await metadataApi.getScienceTypes();
		} catch (err) {
			console.error('Failed to load science types:', err);
		} finally {
			initialLoading = false;
		}

		const cached = getCachedResults();
		if (cached) {
			results = cached;
			statusMessage = `Loaded cached metadata (${cached.count ?? cached.data.length} rows)`;
		}
	});

	async function runQuery(query: MetadataQuery) {
		loading = true;
		error = '';
		try {
			const data = await metadataApi.query(query);
			results = data;
			persistResults(data);
			statusMessage = `Fetched ${data.count} rows from ALMA TAP`;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Unable to fetch metadata';
		} finally {
			loading = false;
		}
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

		loading = true;
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

			results = data;
			persistResults(data);
			loadModalOpen = false;
			formElement.reset();
			statusMessage = `Loaded ${data.count} rows from ${file instanceof File ? 'local file' : filePath}`;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Unable to load metadata file';
		} finally {
			loading = false;
		}
	}

	async function handleSaveMetadata(event: Event) {
		event.preventDefault();
		const current = results;
		if (!current || !current.data?.length) {
			error = 'No metadata to save yet. Run a query or load data first.';
			return;
		}
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
			await metadataApi.save({ path: backendPath, data: current.data });
			backendSucceeded = true;
			statusMessage = `Saved ${current.count ?? current.data.length} rows to backend path ${backendPath}`;
		} catch (err) {
			const message =
				err instanceof Error
					? err.message
					: "Unable to save metadata via API. We'll keep working on a local copy.";
			error = message;
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
					new Blob([JSON.stringify(current, null, 2)], { type: 'application/json' })
				);
				await writable.close();
				statusMessage = `Saved locally to ${localSaveFileName || localSaveHandle.name || 'metadata.json'}${backendSucceeded ? ' (backend copy updated too)' : ''}`;
			} else if (needsLocalFallback) {
				downloadMetadata(current, backendPath);
				statusMessage = `Downloaded metadata snapshot (${current.count ?? current.data.length} rows)`;
			}
		} catch (localError) {
			console.error(localError);
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
		results = null;
		statusMessage = 'Metadata cleared.';
		if (typeof window !== 'undefined') {
			window.localStorage.removeItem(RESULTS_CACHE_KEY);
		}
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

		<MetadataQueryForm {scienceTypes} {loading} onSubmit={runQuery} />

		{#if error}
			<div class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
				{error}
			</div>
		{/if}

		{#if statusMessage}
			<div class="rounded-md border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
				{statusMessage}
			</div>
		{/if}

		<MetadataResultsTable
			{results}
			{loading}
			onClear={clearMetadata}
			onLoad={() => (loadModalOpen = true)}
			onSave={() => (saveModalOpen = true)}
			{saving}
		/>

		<MetadataLoadModal
			open={loadModalOpen}
			{loading}
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
	</div>
</div>
