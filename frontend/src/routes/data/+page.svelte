<script lang="ts">
	import type { DownloadJobSummary, DownloadJobStatus } from '$lib/api/download';
	import { downloadApi } from '$lib/api/download';
	import DownloadProgress from '$lib/components/metadata/DownloadProgress.svelte';
	import { createLogger } from '$lib/logger';

	const logger = createLogger('routes/data');

	let jobs = $state<DownloadJobSummary[]>([]);
	let loading = $state(true);
	let error = $state('');

	// Expanded job detail (for viewing per-file status)
	let expandedJobId = $state<string | null>(null);

	// Re-download state
	let redownloading = $state<Set<string>>(new Set());

	// Poll interval for active jobs
	let pollTimer: ReturnType<typeof setInterval> | undefined;

	const ACTIVE_STATUSES = new Set(['pending', 'starting', 'running', 'downloading']);
	const activeJobs = $derived(jobs.filter((j) => ACTIVE_STATUSES.has(j.status)));
	const hasActiveJobs = $derived(activeJobs.length > 0);

	async function fetchJobs() {
		try {
			jobs = await downloadApi.listJobs();
			error = '';
			logger.debug({ jobCount: jobs.length }, 'Job list refreshed');
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load download history';
			logger.error({ err: e }, 'Failed to fetch download jobs');
		} finally {
			loading = false;
		}
	}

	// Initial load + polling for active jobs
	$effect(() => {
		logger.info('Data page mounted');
		fetchJobs();

		pollTimer = setInterval(fetchJobs, 3000);

		return () => {
			if (pollTimer) clearInterval(pollTimer);
		};
	});

	async function redownload(job: DownloadJobSummary) {
		logger.info({ jobId: job.job_id }, 'Re-download requested');
		const newSet = new Set(redownloading);
		newSet.add(job.job_id);
		redownloading = newSet;
		try {
			await downloadApi.redownloadJob(job.job_id);
			logger.info({ jobId: job.job_id }, 'Re-download job started');
			await fetchJobs();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to start re-download';
			logger.error({ err: e, jobId: job.job_id }, 'Re-download failed');
		} finally {
			const s = new Set(redownloading);
			s.delete(job.job_id);
			redownloading = s;
		}
	}

	async function cancelJob(jobId: string) {
		logger.info({ jobId }, 'Cancel job requested');
		try {
			await downloadApi.cancelJob(jobId);
			logger.info({ jobId }, 'Job cancelled');
			await fetchJobs();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to cancel job';
			logger.error({ err: e, jobId }, 'Failed to cancel job');
		}
	}

	async function deleteJob(jobId: string) {
		logger.info({ jobId }, 'Delete job requested');
		try {
			await downloadApi.deleteJob(jobId);
			logger.info({ jobId }, 'Job deleted from history');
			await fetchJobs();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete job';
			logger.error({ err: e, jobId }, 'Failed to delete job');
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	function formatDate(iso: string): string {
		try {
			return new Date(iso).toLocaleString();
		} catch {
			return iso;
		}
	}

	function statusColor(s: string): string {
		switch (s) {
			case 'running':
			case 'pending':
			case 'starting':
			case 'downloading':
				return 'bg-blue-100 text-blue-700';
			case 'completed':
				return 'bg-green-100 text-green-700';
			case 'failed':
				return 'bg-red-100 text-red-700';
			case 'cancelled':
				return 'bg-gray-100 text-gray-600';
			default:
				return 'bg-gray-100 text-gray-600';
		}
	}

	function progressPercent(progress: number): number {
		if (!Number.isFinite(progress)) return 0;
		return Math.max(0, Math.min(100, Math.round(progress * 100)));
	}
</script>

<div class="space-y-6">
	<header>
		<h1 class="text-3xl font-bold text-gray-900">Data</h1>
		<p class="mt-2 text-gray-600">
			View download history, monitor active downloads, and re-download previous jobs.
		</p>
	</header>

	{#if error}
		<div class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
			{error}
			<button type="button" class="ml-2 underline" onclick={() => (error = '')}>dismiss</button>
		</div>
	{/if}

	<!-- Active downloads summary -->
	{#if hasActiveJobs}
		<div class="rounded-lg border border-blue-200 bg-blue-50 p-4">
			<h2 class="text-sm font-semibold text-blue-800">
				{activeJobs.length} active download{activeJobs.length > 1 ? 's' : ''}
			</h2>
			<div class="mt-3 space-y-3">
				{#each activeJobs as job (job.job_id)}
					<DownloadProgress jobId={job.job_id} onDone={fetchJobs} />
				{/each}
			</div>
		</div>
	{/if}

	<!-- Download history table -->
	<div class="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
		<div class="border-b border-gray-200 px-4 py-3">
			<div class="flex items-center justify-between">
				<h2 class="text-lg font-semibold text-gray-800">Download History</h2>
				<button
					type="button"
					class="rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-50"
					onclick={fetchJobs}
				>
					Refresh
				</button>
			</div>
		</div>

		{#if loading}
			<div class="p-8 text-center text-sm text-gray-400">Loading download history…</div>
		{:else if jobs.length === 0}
			<div class="p-8 text-center text-sm text-gray-400">
				No downloads yet. Start downloading from the
				<a href="/metadata" class="text-blue-600 underline hover:text-blue-700">Metadata</a>
				page.
			</div>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead class="bg-gray-50 text-left text-xs uppercase text-gray-500">
						<tr>
							<th class="px-4 py-3 font-medium">Date</th>
							<th class="px-4 py-3 font-medium">Status</th>
							<th class="px-4 py-3 font-medium">Files</th>
							<th class="px-4 py-3 font-medium">Size</th>
							<th class="px-4 py-3 font-medium">Progress</th>
							<th class="px-4 py-3 font-medium">Destination</th>
							<th class="px-4 py-3 font-medium">Filter</th>
							<th class="px-4 py-3 font-medium text-right">Actions</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-gray-100">
						{#each jobs as job (job.job_id)}
							{@const pct = progressPercent(job.progress)}
							{@const isActive = ACTIVE_STATUSES.has(job.status)}
							<tr class="hover:bg-gray-50" class:bg-blue-50={isActive}>
								<td class="whitespace-nowrap px-4 py-3 text-gray-600">
									{formatDate(job.created_at)}
								</td>
								<td class="px-4 py-3">
									<span
										class="inline-block rounded-full px-2 py-0.5 text-xs font-medium {statusColor(job.status)}"
									>
										{job.status}
									</span>
								</td>
								<td class="px-4 py-3 text-gray-700">
									{job.files_completed}/{job.total_files}
									{#if job.files_failed > 0}
										<span class="text-red-500">({job.files_failed} failed)</span>
									{/if}
								</td>
								<td class="whitespace-nowrap px-4 py-3 text-gray-600">
									{formatBytes(job.total_bytes)}
								</td>
								<td class="px-4 py-3">
									<div class="flex items-center gap-2">
										<div class="h-2 w-20 overflow-hidden rounded-full bg-gray-200">
											<div
												class="h-full rounded-full transition-all duration-300"
												class:bg-blue-500={isActive}
												class:bg-green-500={job.status === 'completed'}
												class:bg-red-500={job.status === 'failed'}
												class:bg-gray-400={job.status === 'cancelled'}
												style="width: {pct}%"
											></div>
										</div>
										<span class="text-xs text-gray-500">{pct}%</span>
									</div>
								</td>
								<td class="max-w-[200px] truncate px-4 py-3 text-gray-500" title={job.destination}>
									{job.destination}
								</td>
								<td class="px-4 py-3">
									<span class="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600">
										{job.product_filter}
									</span>
								</td>
								<td class="whitespace-nowrap px-4 py-3 text-right">
									<div class="flex items-center justify-end gap-2">
										{#if isActive}
											<button
												type="button"
												class="rounded-md border border-red-300 px-2.5 py-1 text-xs font-medium text-red-600 hover:bg-red-50"
												onclick={() => cancelJob(job.job_id)}
											>
												Cancel
											</button>
										{:else}
											{#if job.error}
												<span class="text-xs text-red-500" title={job.error}>⚠</span>
											{/if}
											<button
												type="button"
												class="rounded-md border border-blue-300 px-2.5 py-1 text-xs font-medium text-blue-600 hover:bg-blue-50 disabled:opacity-50"
												disabled={redownloading.has(job.job_id)}
												title="Download these products again (overwrites existing files)"
												onclick={() => redownload(job)}
											>
												{redownloading.has(job.job_id) ? 'Starting…' : 'Re-download'}
											</button>
											<button
												type="button"
												class="rounded-md border border-gray-300 px-2.5 py-1 text-xs font-medium text-gray-500 hover:bg-red-50 hover:text-red-600 hover:border-red-300"
												title="Remove from history"
												onclick={() => deleteJob(job.job_id)}
											>
												Delete
											</button>
										{/if}
									</div>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>
</div>
