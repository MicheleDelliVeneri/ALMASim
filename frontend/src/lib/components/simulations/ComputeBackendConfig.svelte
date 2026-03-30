<script lang="ts">
	import { simulationApi, type DaskTestResult } from '$lib/api/simulation';

	interface Props {
		backendType: string;
		backendConfig: Record<string, unknown>;
		onBackendTypeChange: (type: string) => void;
		onConfigChange: (config: Record<string, unknown>) => void;
	}

	let { backendType, backendConfig, onBackendTypeChange, onConfigChange }: Props = $props();

	// Dask connection test state
	let daskTesting = $state(false);
	let daskResult: DaskTestResult | null = $state(null);
	let daskTestError: string | null = $state(null);

	// Warn when the user accidentally enters the dashboard port (8787) instead of scheduler port (8786)
	const daskSchedulerAddr = $derived((backendConfig.scheduler as string) || '');
	const daskPortWarning = $derived(/:8787\b/.test(daskSchedulerAddr)
		? 'Port 8787 is the Dask dashboard — the scheduler typically listens on port 8786.'
		: null);
	// Warn when localhost is used — the backend runs inside Docker so localhost resolves to the container, not the host
	const daskLocalhostWarning = $derived(
		/\/\/localhost\b/.test(daskSchedulerAddr)
			? 'The backend runs inside Docker — use host.docker.internal instead of localhost (e.g. tcp://host.docker.internal:8786).'
			: null
	);

	function updateConfig(key: string, value: unknown) {
		onConfigChange({ ...backendConfig, [key]: value });
	}

	function removeConfigKey(key: string) {
		const { [key]: _, ...rest } = backendConfig;
		onConfigChange(rest);
	}

	async function testDaskConnection() {
		const scheduler = (backendConfig.scheduler as string) || '';
		if (!scheduler) {
			daskTestError = 'Enter a scheduler address first';
			daskResult = null;
			return;
		}
		daskTesting = true;
		daskTestError = null;
		daskResult = null;
		try {
			daskResult = await simulationApi.testDask(scheduler);
			if (!daskResult.ok) {
				daskTestError = daskResult.error || 'Connection failed';
			}
		} catch (e: unknown) {
			daskTestError = e instanceof Error ? e.message : 'Connection failed';
		} finally {
			daskTesting = false;
		}
	}

	function openDaskDashboard() {
		if (!daskResult?.ok || !daskResult.dashboard_port) return;
		// Extract host from the scheduler address the user typed
		const addr = (backendConfig.scheduler as string) || '';
		let host = addr.replace(/^tcp:\/\//, '').split(':')[0] || 'localhost';
		// Docker-internal hostnames aren't reachable from the browser
		if (host === 'dask-scheduler' || host.includes('.internal') || !host.includes('.')) {
			host = 'localhost';
		}
		window.open(`http://${host}:${daskResult.dashboard_port}/status`, '_blank');
	}
</script>

<section class="space-y-4 rounded-lg bg-white p-6 shadow-md">
	<h2 class="text-xl font-semibold text-gray-900">Compute Backend</h2>
	<p class="text-sm text-gray-600">Select the computation backend and configure its parameters.</p>

	<div class="space-y-4">
		<div>
			<label for="compute_backend" class="mb-1 block text-sm font-medium text-gray-700">
				Backend Type
			</label>
			<select
				id="compute_backend"
				value={backendType}
				onchange={(e) => {
					onBackendTypeChange(e.currentTarget.value);
					onConfigChange({}); // Reset config when backend changes
				}}
				class="w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
			>
				<option value="local">Local</option>
				<option value="dask">Dask (Distributed)</option>
				<option value="slurm">Slurm (HPC Cluster)</option>
				<option value="kubernetes">Kubernetes</option>
			</select>
		</div>

		{#if backendType === 'local'}
			<div class="space-y-3 rounded-md border border-gray-200 bg-gray-50 p-4">
				<h3 class="text-sm font-semibold text-gray-800">Local Backend Configuration</h3>
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label for="local_n_workers" class="mb-1 block text-xs font-medium text-gray-700">
							Number of Worker Processes
						</label>
						<input
							type="number"
							id="local_n_workers"
							min="1"
							value={(backendConfig.n_workers as number) || ''}
							oninput={(e) => {
								const val = e.currentTarget.value.trim();
								if (val === '') {
									removeConfigKey('n_workers');
								} else {
									const numVal = parseInt(val);
									updateConfig('n_workers', isNaN(numVal) ? undefined : numVal);
								}
							}}
							placeholder="Auto (CPU count)"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
				</div>
			</div>
		{:else if backendType === 'dask'}
			<div class="space-y-3 rounded-md border border-gray-200 bg-gray-50 p-4">
				<h3 class="text-sm font-semibold text-gray-800">Dask Backend Configuration</h3>
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label for="dask_scheduler" class="mb-1 block text-xs font-medium text-gray-700">
							Scheduler Address
						</label>
						<div class="flex gap-2">
							<input
								type="text"
								id="dask_scheduler"
								value={(backendConfig.scheduler as string) || ''}
								oninput={(e) => {
									const val = e.currentTarget.value.trim();
									updateConfig('scheduler', val || undefined);
									// Reset test state when address changes
									daskResult = null;
									daskTestError = null;
								}}
								placeholder="tcp://localhost:8786"
								class="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
							/>
							<button
								type="button"
								onclick={testDaskConnection}
								disabled={daskTesting}
								class="inline-flex items-center gap-1 rounded-md bg-blue-600 px-3 py-2 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
								title="Test connection"
							>
								{#if daskTesting}
									<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
										<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25" />
										<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
									</svg>
								{:else}
									Test
								{/if}
							</button>
						</div>
						<p class="mt-1 text-xs text-gray-500">
							Scheduler port is <span class="font-mono">8786</span>; dashboard is on <span class="font-mono">8787</span>. Leave empty for a local cluster.
						</p>

						{#if daskPortWarning}
							<div class="mt-1 flex items-center gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
								<svg class="h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
									<path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"/>
								</svg>
								{daskPortWarning}
							</div>
						{/if}

						{#if daskLocalhostWarning}
							<div class="mt-1 flex items-start gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
								<svg class="mt-0.5 h-3.5 w-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
									<path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"/>
								</svg>
								{daskLocalhostWarning}
							</div>
						{/if}

						<!-- Connection test result -->
						{#if daskResult?.ok}
							<div class="mt-2 flex items-center gap-2 rounded-md border border-green-200 bg-green-50 px-3 py-2">
								<span class="inline-block h-2.5 w-2.5 rounded-full bg-green-500"></span>
								<span class="text-xs font-medium text-green-800">
									Connected — {daskResult.workers} worker{daskResult.workers !== 1 ? 's' : ''}, {daskResult.total_threads} threads, {daskResult.total_memory_gb} GB
								</span>
								<button
									type="button"
									onclick={openDaskDashboard}
									class="ml-auto inline-flex items-center gap-1 rounded bg-orange-100 px-2 py-1 text-xs font-medium text-orange-700 hover:bg-orange-200"
									title="Open Dask Dashboard"
								>
									<!-- Dask logo (simplified) -->
									<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
										<path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
										<path d="M12 22V12" stroke="currentColor" stroke-width="2"/>
										<path d="M3 7l9 5 9-5" stroke="currentColor" stroke-width="2"/>
									</svg>
									Dashboard
								</button>
							</div>
						{:else if daskTestError}
							<div class="mt-2 rounded-md border border-red-200 bg-red-50 px-3 py-2">
								<div class="flex items-center gap-2">
									<span class="inline-block h-2.5 w-2.5 shrink-0 rounded-full bg-red-500"></span>
									<span class="text-xs font-medium text-red-800">{daskTestError}</span>
								</div>
								{#if /localhost/.test(daskSchedulerAddr) && /timeout|refused|connect/i.test(daskTestError)}
									<p class="mt-1 pl-4 text-xs text-red-700">
										Tip: try <span class="font-mono font-semibold">tcp://host.docker.internal:8786</span> — the backend runs inside Docker and cannot reach your host via <span class="font-mono">localhost</span>.
									</p>
								{/if}
							</div>
						{/if}
					</div>
					<div>
						<label for="dask_n_workers" class="mb-1 block text-xs font-medium text-gray-700">
							Number of Workers
						</label>
						<input
							type="number"
							id="dask_n_workers"
							min="1"
							value={(backendConfig.n_workers as number) || ''}
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value);
								updateConfig('n_workers', isNaN(val) ? undefined : val);
							}}
							placeholder="Auto"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
						<p class="mt-1 text-xs text-gray-500">
							Only used for local Dask cluster (ignored when using external scheduler).
						</p>
					</div>
				</div>
			</div>
		{:else if backendType === 'slurm'}
			<div class="space-y-3 rounded-md border border-gray-200 bg-gray-50 p-4">
				<h3 class="text-sm font-semibold text-gray-800">Slurm Backend Configuration</h3>
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label for="slurm_queue" class="mb-1 block text-xs font-medium text-gray-700">
							Queue Name
						</label>
						<input
							type="text"
							id="slurm_queue"
							value={(backendConfig.queue as string) || 'normal'}
							oninput={(e) => updateConfig('queue', e.currentTarget.value)}
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="slurm_project" class="mb-1 block text-xs font-medium text-gray-700">
							Project/Account (optional)
						</label>
						<input
							type="text"
							id="slurm_project"
							value={(backendConfig.project as string) || ''}
							oninput={(e) => {
								const val = e.currentTarget.value.trim();
								updateConfig('project', val || undefined);
							}}
							placeholder="Optional"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="slurm_walltime" class="mb-1 block text-xs font-medium text-gray-700">
							Walltime (HH:MM:SS)
						</label>
						<input
							type="text"
							id="slurm_walltime"
							value={(backendConfig.walltime as string) || '02:00:00'}
							oninput={(e) => updateConfig('walltime', e.currentTarget.value)}
							placeholder="02:00:00"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="slurm_cores" class="mb-1 block text-xs font-medium text-gray-700">
							Cores per Worker
						</label>
						<input
							type="number"
							id="slurm_cores"
							min="1"
							value={(backendConfig.cores as number) || 1}
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value);
								updateConfig('cores', isNaN(val) ? 1 : val);
							}}
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="slurm_memory" class="mb-1 block text-xs font-medium text-gray-700">
							Memory per Worker
						</label>
						<input
							type="text"
							id="slurm_memory"
							value={(backendConfig.memory as string) || '4GB'}
							oninput={(e) => updateConfig('memory', e.currentTarget.value)}
							placeholder="4GB"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="slurm_n_workers" class="mb-1 block text-xs font-medium text-gray-700">
							Number of Workers
						</label>
						<input
							type="number"
							id="slurm_n_workers"
							min="1"
							value={(backendConfig.n_workers as number) || 4}
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value);
								updateConfig('n_workers', isNaN(val) ? 4 : val);
							}}
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
				</div>
			</div>
		{:else if backendType === 'kubernetes'}
			<div class="space-y-3 rounded-md border border-gray-200 bg-gray-50 p-4">
				<h3 class="text-sm font-semibold text-gray-800">Kubernetes Backend Configuration</h3>
				<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
					<div>
						<label for="k8s_namespace" class="mb-1 block text-xs font-medium text-gray-700">
							Namespace (optional)
						</label>
						<input
							type="text"
							id="k8s_namespace"
							value={(backendConfig.namespace as string) || ''}
							oninput={(e) => {
								const val = e.currentTarget.value.trim();
								updateConfig('namespace', val || undefined);
							}}
							placeholder="default"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="k8s_n_workers" class="mb-1 block text-xs font-medium text-gray-700">
							Number of Workers
						</label>
						<input
							type="number"
							id="k8s_n_workers"
							min="1"
							value={(backendConfig.n_workers as number) || 4}
							oninput={(e) => {
								const val = parseInt(e.currentTarget.value);
								updateConfig('n_workers', isNaN(val) ? 4 : val);
							}}
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="k8s_image" class="mb-1 block text-xs font-medium text-gray-700">
							Docker Image (optional)
						</label>
						<input
							type="text"
							id="k8s_image"
							value={(backendConfig.image as string) || ''}
							oninput={(e) => {
								const val = e.currentTarget.value.trim();
								updateConfig('image', val || undefined);
							}}
							placeholder="daskdev/dask:latest"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
					</div>
					<div>
						<label for="k8s_resources" class="mb-1 block text-xs font-medium text-gray-700">
							Resources (JSON, optional)
						</label>
						<textarea
							id="k8s_resources"
							rows={3}
							value={JSON.stringify(backendConfig.resources || {}, null, 2)}
							oninput={(e) => {
								try {
									const val = e.currentTarget.value.trim();
									const parsed = val ? JSON.parse(val) : {};
									updateConfig('resources', parsed);
								} catch {
									// Invalid JSON, ignore
								}
							}}
							placeholder={`{"requests": {"cpu": "1", "memory": "2Gi"}}`}
							class="w-full rounded-md border border-gray-300 px-3 py-2 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						></textarea>
						<p class="mt-1 text-xs text-gray-500">JSON format for resource requests/limits</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
</section>
