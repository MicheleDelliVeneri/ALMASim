interface DatacubeUploadProps {
  onFileUpload: (event: Event) => void;
  integrationMethod: "sum" | "mean";
  onMethodChange: (method: "sum" | "mean") => void;
  loading: boolean;
}

export function DatacubeUpload(props: DatacubeUploadProps) {
  return (
    <div class="space-y-4">
      {/* Divider */}
      <div class="relative">
        <div class="absolute inset-0 flex items-center">
          <div class="w-full border-t border-gray-300"></div>
        </div>
        <div class="relative flex justify-center text-sm">
          <span class="px-2 bg-white text-gray-500">OR</span>
        </div>
      </div>

      {/* File Upload */}
      <div class="space-y-2">
        <label class="block text-sm font-medium text-gray-700">
          Upload Datacube (.npz file)
        </label>
        <div class="flex items-center space-x-4">
          <input
            type="file"
            accept=".npz"
            onChange={props.onFileUpload}
            disabled={props.loading}
            class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          />
        </div>
      </div>

      {/* Integration Method */}
      <div class="space-y-2">
        <label class="block text-sm font-medium text-gray-700">
          Integration Method
        </label>
        <select
          value={props.integrationMethod}
          onChange={(e) => props.onMethodChange(e.currentTarget.value as "sum" | "mean")}
          disabled={props.loading}
          class="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          <option value="sum">Sum (integrate all channels)</option>
          <option value="mean">Mean (average over channels)</option>
        </select>
      </div>
    </div>
  );
}

