import { For, Show } from "solid-js";

interface DatacubeFile {
  name: string;
  path: string;
  size: number;
  modified: number;
}

interface DatacubeFileListProps {
  files: DatacubeFile[];
  loading: boolean;
  outputDir: string;
  onFileSelect: (path: string) => void;
  onRefresh: () => void;
  disabled: boolean;
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatDate = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleString();
};

export function DatacubeFileList(props: DatacubeFileListProps) {
  return (
    <div class="space-y-4">
      {/* Output Directory Info */}
      <Show when={props.outputDir}>
        <div class="bg-blue-50 border border-blue-200 rounded-md p-3">
          <p class="text-sm text-blue-800">
            <span class="font-medium">Output Directory:</span> {props.outputDir}
          </p>
        </div>
      </Show>

      {/* Available Files List */}
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <label class="block text-sm font-medium text-gray-700">
            Available Datacubes
          </label>
          <button
            onClick={props.onRefresh}
            class="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            disabled={props.loading}
          >
            {props.loading ? "Loading..." : "Refresh"}
          </button>
        </div>
        <Show
          when={!props.loading && props.files.length > 0}
          fallback={
            <div class="text-sm text-gray-500 py-4 text-center border border-gray-200 rounded-md">
              {props.loading ? "Loading files..." : "No .npz files found in output directory"}
            </div>
          }
        >
          <div class="border border-gray-200 rounded-md max-h-64 overflow-y-auto">
            <table class="min-w-full divide-y divide-gray-200">
              <thead class="bg-gray-50 sticky top-0">
                <tr>
                  <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    File Name
                  </th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Modified
                  </th>
                  <th class="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody class="bg-white divide-y divide-gray-200">
                <For each={props.files}>
                  {(file) => (
                    <tr class="hover:bg-gray-50">
                      <td class="px-4 py-2 text-sm text-gray-900 font-mono">
                        {file.name}
                      </td>
                      <td class="px-4 py-2 text-sm text-gray-600">
                        {formatFileSize(file.size)}
                      </td>
                      <td class="px-4 py-2 text-sm text-gray-600">
                        {formatDate(file.modified)}
                      </td>
                      <td class="px-4 py-2 text-right">
                        <button
                          onClick={() => props.onFileSelect(file.path)}
                          disabled={props.disabled}
                          class="px-3 py-1 text-xs bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                          Load
                        </button>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </Show>
      </div>
    </div>
  );
}

