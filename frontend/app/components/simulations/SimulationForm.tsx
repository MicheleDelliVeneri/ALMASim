interface SimulationFormProps {
  sourceType: string;
  nPix: number;
  nChannels: number;
  onSourceTypeChange: (type: string) => void;
  onNPixChange: (value: number) => void;
  onNChannelsChange: (value: number) => void;
}

export function SimulationForm(props: SimulationFormProps) {
  return (
    <section class="bg-white rounded-lg shadow-md p-6 space-y-4">
      <h2 class="text-xl font-semibold text-gray-900">Simulation Configuration</h2>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label for="source_type" class="block text-sm font-medium text-gray-700 mb-1">
            Source Type
          </label>
          <select
            id="source_type"
            value={props.sourceType}
            onChange={(e) => props.onSourceTypeChange(e.currentTarget.value)}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="point">Point</option>
            <option value="gaussian">Gaussian</option>
            <option value="extended">Extended</option>
            <option value="diffuse">Diffuse</option>
            <option value="galaxy-zoo">Galaxy Zoo</option>
            <option value="molecular">Molecular</option>
            <option value="hubble-100">Hubble</option>
          </select>
        </div>

        <div>
          <label for="n_pix" class="block text-sm font-medium text-gray-700 mb-1">
            Spatial Pixels (n_pix × n_pix)
          </label>
          <input
            type="number"
            id="n_pix"
            min="32"
            max="2048"
            step="32"
            value={props.nPix}
            onInput={(e) => props.onNPixChange(parseInt(e.currentTarget.value) || 256)}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p class="text-xs text-gray-500 mt-1">Cube will be {props.nPix} × {props.nPix} × {props.nChannels}</p>
        </div>

        <div>
          <label for="n_channels" class="block text-sm font-medium text-gray-700 mb-1">
            Spectral Channels
          </label>
          <input
            type="number"
            id="n_channels"
            min="16"
            max="1024"
            step="16"
            value={props.nChannels}
            onInput={(e) => props.onNChannelsChange(parseInt(e.currentTarget.value) || 128)}
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p class="text-xs text-gray-500 mt-1">Total channels: {props.nChannels}</p>
        </div>
      </div>
    </section>
  );
}

