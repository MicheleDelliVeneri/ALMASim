import { Show } from "solid-js";

interface ComputeBackendConfigProps {
  backendType: string;
  backendConfig: Record<string, unknown>;
  onBackendTypeChange: (type: string) => void;
  onConfigChange: (config: Record<string, unknown>) => void;
}

export function ComputeBackendConfig(props: ComputeBackendConfigProps) {
  const updateConfig = (key: string, value: unknown) => {
    props.onConfigChange({ ...props.backendConfig, [key]: value });
  };

  const removeConfigKey = (key: string) => {
    const { [key]: _, ...rest } = props.backendConfig;
    props.onConfigChange(rest);
  };

  return (
    <section class="bg-white rounded-lg shadow-md p-6 space-y-4">
      <h2 class="text-xl font-semibold text-gray-900">Compute Backend</h2>
      <p class="text-sm text-gray-600">
        Select the computation backend and configure its parameters.
      </p>
      
      <div class="space-y-4">
        <div>
          <label for="compute_backend" class="block text-sm font-medium text-gray-700 mb-1">
            Backend Type
          </label>
          <select
            id="compute_backend"
            value={props.backendType}
            onChange={(e) => {
              props.onBackendTypeChange(e.currentTarget.value);
              props.onConfigChange({}); // Reset config when backend changes
            }}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="local">Local</option>
            <option value="dask">Dask (Distributed)</option>
            <option value="slurm">Slurm (HPC Cluster)</option>
            <option value="kubernetes">Kubernetes</option>
          </select>
        </div>

        {/* Local Backend Config */}
        <Show when={props.backendType === "local"}>
          <div class="bg-gray-50 rounded-md p-4 space-y-3 border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-800">Local Backend Configuration</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="local_n_workers" class="block text-xs font-medium text-gray-700 mb-1">
                  Number of Worker Processes
                </label>
                <input
                  type="number"
                  id="local_n_workers"
                  min="1"
                  value={(props.backendConfig.n_workers as number) || ""}
                  onInput={(e) => {
                    const val = e.currentTarget.value.trim();
                    if (val === "") {
                      removeConfigKey("n_workers");
                    } else {
                      const numVal = parseInt(val);
                      updateConfig("n_workers", isNaN(numVal) ? undefined : numVal);
                    }
                  }}
                  placeholder="Auto (CPU count)"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </Show>

        {/* Dask Backend Config */}
        <Show when={props.backendType === "dask"}>
          <div class="bg-gray-50 rounded-md p-4 space-y-3 border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-800">Dask Backend Configuration</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="dask_scheduler" class="block text-xs font-medium text-gray-700 mb-1">
                  Scheduler Address (optional)
                </label>
                <input
                  type="text"
                  id="dask_scheduler"
                  value={(props.backendConfig.scheduler as string) || ""}
                  onInput={(e) => {
                    const val = e.currentTarget.value.trim();
                    updateConfig("scheduler", val || undefined);
                  }}
                  placeholder="tcp://localhost:8786"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p class="text-xs text-gray-500 mt-1">Leave empty for local Dask cluster (uses processes)</p>
              </div>
              <div>
                <label for="dask_n_workers" class="block text-xs font-medium text-gray-700 mb-1">
                  Number of Workers
                </label>
                <input
                  type="number"
                  id="dask_n_workers"
                  min="1"
                  value={(props.backendConfig.n_workers as number) || ""}
                  onInput={(e) => {
                    const val = parseInt(e.currentTarget.value);
                    updateConfig("n_workers", isNaN(val) ? undefined : val);
                  }}
                  placeholder="Auto"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                  Local Dask cluster uses processes for true parallelism.
                </p>
              </div>
            </div>
          </div>
        </Show>

        {/* Slurm Backend Config */}
        <Show when={props.backendType === "slurm"}>
          <div class="bg-gray-50 rounded-md p-4 space-y-3 border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-800">Slurm Backend Configuration</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="slurm_queue" class="block text-xs font-medium text-gray-700 mb-1">
                  Queue Name
                </label>
                <input
                  type="text"
                  id="slurm_queue"
                  value={(props.backendConfig.queue as string) || "normal"}
                  onInput={(e) => updateConfig("queue", e.currentTarget.value)}
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="slurm_project" class="block text-xs font-medium text-gray-700 mb-1">
                  Project/Account (optional)
                </label>
                <input
                  type="text"
                  id="slurm_project"
                  value={(props.backendConfig.project as string) || ""}
                  onInput={(e) => {
                    const val = e.currentTarget.value.trim();
                    updateConfig("project", val || undefined);
                  }}
                  placeholder="Optional"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="slurm_walltime" class="block text-xs font-medium text-gray-700 mb-1">
                  Walltime (HH:MM:SS)
                </label>
                <input
                  type="text"
                  id="slurm_walltime"
                  value={(props.backendConfig.walltime as string) || "02:00:00"}
                  onInput={(e) => updateConfig("walltime", e.currentTarget.value)}
                  placeholder="02:00:00"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="slurm_cores" class="block text-xs font-medium text-gray-700 mb-1">
                  Cores per Worker
                </label>
                <input
                  type="number"
                  id="slurm_cores"
                  min="1"
                  value={(props.backendConfig.cores as number) || 1}
                  onInput={(e) => {
                    const val = parseInt(e.currentTarget.value);
                    updateConfig("cores", isNaN(val) ? 1 : val);
                  }}
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="slurm_memory" class="block text-xs font-medium text-gray-700 mb-1">
                  Memory per Worker
                </label>
                <input
                  type="text"
                  id="slurm_memory"
                  value={(props.backendConfig.memory as string) || "4GB"}
                  onInput={(e) => updateConfig("memory", e.currentTarget.value)}
                  placeholder="4GB"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="slurm_n_workers" class="block text-xs font-medium text-gray-700 mb-1">
                  Number of Workers
                </label>
                <input
                  type="number"
                  id="slurm_n_workers"
                  min="1"
                  value={(props.backendConfig.n_workers as number) || 4}
                  onInput={(e) => {
                    const val = parseInt(e.currentTarget.value);
                    updateConfig("n_workers", isNaN(val) ? 4 : val);
                  }}
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </Show>

        {/* Kubernetes Backend Config */}
        <Show when={props.backendType === "kubernetes"}>
          <div class="bg-gray-50 rounded-md p-4 space-y-3 border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-800">Kubernetes Backend Configuration</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="k8s_namespace" class="block text-xs font-medium text-gray-700 mb-1">
                  Namespace (optional)
                </label>
                <input
                  type="text"
                  id="k8s_namespace"
                  value={(props.backendConfig.namespace as string) || ""}
                  onInput={(e) => {
                    const val = e.currentTarget.value.trim();
                    updateConfig("namespace", val || undefined);
                  }}
                  placeholder="default"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="k8s_n_workers" class="block text-xs font-medium text-gray-700 mb-1">
                  Number of Workers
                </label>
                <input
                  type="number"
                  id="k8s_n_workers"
                  min="1"
                  value={(props.backendConfig.n_workers as number) || 4}
                  onInput={(e) => {
                    const val = parseInt(e.currentTarget.value);
                    updateConfig("n_workers", isNaN(val) ? 4 : val);
                  }}
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="k8s_image" class="block text-xs font-medium text-gray-700 mb-1">
                  Docker Image (optional)
                </label>
                <input
                  type="text"
                  id="k8s_image"
                  value={(props.backendConfig.image as string) || ""}
                  onInput={(e) => {
                    const val = e.currentTarget.value.trim();
                    updateConfig("image", val || undefined);
                  }}
                  placeholder="daskdev/dask:latest"
                  class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label for="k8s_resources" class="block text-xs font-medium text-gray-700 mb-1">
                  Resources (JSON, optional)
                </label>
                <textarea
                  id="k8s_resources"
                  rows={3}
                  value={JSON.stringify(props.backendConfig.resources || {}, null, 2)}
                  onInput={(e) => {
                    try {
                      const val = e.currentTarget.value.trim();
                      const parsed = val ? JSON.parse(val) : {};
                      updateConfig("resources", parsed);
                    } catch {
                      // Invalid JSON, ignore
                    }
                  }}
                  placeholder='{"requests": {"cpu": "1", "memory": "2Gi"}}'
                  class="w-full px-3 py-2 text-sm font-mono border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p class="text-xs text-gray-500 mt-1">JSON format for resource requests/limits</p>
              </div>
            </div>
          </div>
        </Show>
      </div>
    </section>
  );
}

