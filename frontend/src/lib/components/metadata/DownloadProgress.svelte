<script lang="ts">
	import type { DownloadJobStatus } from '$lib/api/download';
	import { downloadApi } from '$lib/api/download';

	interface Props {
		jobId: string;
		onDone: () => void;
	}

	let { jobId, onDone }: Props = $props();

	let status = $state<DownloadJobStatus | null>(null);
	let error = $state('');
	let cancelling = $state(false);

	const ACTIVE_STATUSES = new Set(['pending', 'starting', 'running', 'downloading']);
	const progressPct = $derived(status ? progressPercent(status.progress) : 0);
	const isRunning = $derived(status ? ACTIVE_STATUSES.has(status.status) : false);
	const isDone = $derived(
		status?.status === 'completed' ||
			status?.status === 'failed' ||
			status?.status === 'cancelled'
	);

	// --- Speed & ETA tracking ---
	// Keep a rolling window of (timestamp, bytes_downloaded) samples
	const SPEED_WINDOW = 10; // use last N samples for smoothed speed
	let speedSamples = $state<{ time: number; bytes: number }[]>([]);
	let speedBytesPerSec = $state(0);
	let etaSeconds = $state<number | null>(null);

	function updateSpeedAndEta(bytesDownloaded: number, totalBytes: number) {
		const now = Date.now();
		speedSamples = [...speedSamples.slice(-(SPEED_WINDOW - 1)), { time: now, bytes: bytesDownloaded }];

		if (speedSamples.length >= 2) {
			const oldest = speedSamples[0];
			const newest = speedSamples[speedSamples.length - 1];
			const dtSec = (newest.time - oldest.time) / 1000;
			const dBytes = newest.bytes - oldest.bytes;
			if (dtSec > 0 && dBytes > 0) {
				speedBytesPerSec = dBytes / dtSec;
				const remaining = totalBytes - bytesDownloaded;
				etaSeconds = remaining > 0 ? remaining / speedBytesPerSec : 0;
			} else {
				speedBytesPerSec = 0;
				etaSeconds = null;
			}
		}
	}

	// Poll every 2s while running
	$effect(() => {
		if (!jobId) return;
		let timer: ReturnType<typeof setInterval>;
		let stopped = false;

		async function poll() {
			try {
				const s = await downloadApi.getJobStatus(jobId);
				if (stopped) return;
				status = s;
				if (ACTIVE_STATUSES.has(s.status)) {
					updateSpeedAndEta(s.bytes_downloaded, s.total_bytes);
				}
				if (s.status === 'completed' || s.status === 'failed' || s.status === 'cancelled') {
					clearInterval(timer);
					speedBytesPerSec = 0;
					etaSeconds = null;
				}
			} catch (e) {
				if (stopped) return;
				error = e instanceof Error ? e.message : 'Failed to fetch status';
				clearInterval(timer);
			}
		}

		poll();
		timer = setInterval(poll, 2000);

		return () => {
			stopped = true;
			clearInterval(timer);
		};
	});

	async function cancelDownload() {
		cancelling = true;
		try {
			await downloadApi.cancelJob(jobId);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to cancel';
		} finally {
			cancelling = false;
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	function formatSpeed(bytesPerSec: number): string {
		if (bytesPerSec < 1024) return `${bytesPerSec.toFixed(0)} B/s`;
		if (bytesPerSec < 1024 * 1024) return `${(bytesPerSec / 1024).toFixed(1)} KB/s`;
		if (bytesPerSec < 1024 * 1024 * 1024) return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
		return `${(bytesPerSec / (1024 * 1024 * 1024)).toFixed(2)} GB/s`;
	}

	function formatEta(seconds: number): string {
		if (seconds < 60) return `${Math.ceil(seconds)}s`;
		if (seconds < 3600) {
			const m = Math.floor(seconds / 60);
			const s = Math.ceil(seconds % 60);
			return `${m}m ${s}s`;
		}
		const h = Math.floor(seconds / 3600);
		const m = Math.ceil((seconds % 3600) / 60);
		return `${h}h ${m}m`;
	}

	function progressPercent(progress: number): number {
		if (!Number.isFinite(progress)) return 0;
		return Math.max(0, Math.min(100, Math.round(progress * 100)));
	}
</script>

<div class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
	<div class="flex items-center justify-between mb-3">
		<h3 class="text-sm font-semibold text-gray-800">
			Download
			{#if status}
				<span
					class="ml-2 inline-block rounded-full px-2 py-0.5 text-xs font-medium"
					class:bg-blue-100={isRunning}
					class:text-blue-700={isRunning}
					class:bg-green-100={status.status === 'completed'}
					class:text-green-700={status.status === 'completed'}
					class:bg-red-100={status.status === 'failed'}
					class:text-red-700={status.status === 'failed'}
					class:bg-gray-100={status.status === 'cancelled'}
					class:text-gray-600={status.status === 'cancelled'}
				>
					{status.status}
				</span>
			{/if}
		</h3>
		<div class="flex gap-2">
			{#if isRunning}
				<button
					type="button"
					class="rounded-md border border-red-300 px-3 py-1 text-xs font-medium text-red-600 hover:bg-red-50"
					disabled={cancelling}
					onclick={cancelDownload}
				>
					{cancelling ? 'Cancelling…' : 'Cancel'}
				</button>
			{/if}
			{#if isDone}
				<button
					type="button"
					class="rounded-md border border-gray-300 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50"
					onclick={onDone}
				>
					Dismiss
				</button>
			{/if}
		</div>
	</div>

	{#if error}
		<p class="mb-2 text-sm text-red-600">{error}</p>
	{/if}

	{#if status}
		<!-- Overall progress bar -->
		<div class="mb-2">
			<div class="flex justify-between text-xs text-gray-500 mb-1">
				<span>{status.files_completed} / {status.total_files} files</span>
				<span>{progressPct}%</span>
			</div>
			<div class="h-2 w-full overflow-hidden rounded-full bg-gray-200">
				<div
					class="h-full rounded-full transition-all duration-300"
					class:bg-blue-500={isRunning}
					class:bg-green-500={status.status === 'completed'}
					class:bg-red-500={status.status === 'failed'}
					class:bg-gray-400={status.status === 'cancelled'}
					style="width: {progressPct}%"
				></div>
			</div>
			<p class="mt-1 text-xs text-gray-400">
				{formatBytes(status.bytes_downloaded)} / {formatBytes(status.total_bytes)}
				{#if isRunning && speedBytesPerSec > 0}
					<span class="mx-1">·</span>
					<span class="text-blue-500">{formatSpeed(speedBytesPerSec)}</span>
					{#if etaSeconds != null && etaSeconds > 0}
						<span class="mx-1">·</span>
						<span class="text-gray-500">~{formatEta(etaSeconds)} remaining</span>
					{/if}
				{/if}
				{#if status.files_failed > 0}
					<span class="text-red-500">({status.files_failed} failed)</span>
				{/if}
			</p>
		</div>

		<!-- Per-file list (scrollable, compact) -->
		{#if status.files.length > 0}
			<div class="max-h-40 overflow-y-auto border-t border-gray-100 pt-2">
				{#each status.files as file}
					<div class="flex items-center gap-2 py-1 text-xs">
						{#if file.status === 'completed'}
							<span class="text-green-500" title="Done">✓</span>
						{:else if file.status === 'downloading'}
							<span class="text-blue-500" title="Downloading">↓</span>
						{:else if file.status === 'failed'}
							<span class="text-red-500" title={file.error ?? 'Failed'}>✗</span>
						{:else}
							<span class="text-gray-300" title="Pending">○</span>
						{/if}
						<span class="flex-1 truncate text-gray-600" title={file.filename}>
							{file.filename}
						</span>
						<span class="text-gray-400">
							{progressPercent(file.progress)}%
						</span>
					</div>
				{/each}
			</div>
		{/if}

		{#if status.error}
			<p class="mt-2 text-xs text-red-500">{status.error}</p>
		{/if}
	{:else if !error}
		<p class="text-sm text-gray-400">Loading…</p>
	{/if}
</div>
