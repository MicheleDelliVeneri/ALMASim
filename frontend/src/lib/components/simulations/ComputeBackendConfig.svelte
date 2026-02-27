<script lang="ts">
	interface Props {
		backendType: string;
		backendConfig: Record<string, unknown>;
		onBackendTypeChange: (type: string) => void;
		onConfigChange: (config: Record<string, unknown>) => void;
	}

	let { backendType, backendConfig, onBackendTypeChange, onConfigChange }: Props = $props();

	function updateConfig(key: string, value: unknown) {
		onConfigChange({ ...backendConfig, [key]: value });
	}

	function removeConfigKey(key: string) {
		const { [key]: _, ...rest } = backendConfig;
		onConfigChange(rest);
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
							Scheduler Address (optional)
						</label>
						<input
							type="text"
							id="dask_scheduler"
							value={(backendConfig.scheduler as string) || ''}
							oninput={(e) => {
								const val = e.currentTarget.value.trim();
								updateConfig('scheduler', val || undefined);
							}}
							placeholder="tcp://localhost:8786"
							class="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
						<p class="mt-1 text-xs text-gray-500">
							Leave empty for local Dask cluster (uses processes)
						</p>
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
							Local Dask cluster uses processes for true parallelism.
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
