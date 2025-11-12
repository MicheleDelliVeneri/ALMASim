interface ImageStatisticsProps {
  stats: {
    shape: number[];
    integrated_shape: number[];
    min: number;
    max: number;
    mean: number;
    std: number;
    cube_name: string;
  };
  method: string;
}

export function ImageStatistics(props: ImageStatisticsProps) {
  return (
    <div class="bg-gray-50 rounded-md p-4 space-y-2">
      <h3 class="text-sm font-semibold text-gray-900">Datacube Statistics</h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p class="text-gray-600">Original Shape</p>
          <p class="font-mono text-gray-900">
            {props.stats.shape.join(" × ")}
          </p>
        </div>
        <div>
          <p class="text-gray-600">Integrated Shape</p>
          <p class="font-mono text-gray-900">
            {props.stats.integrated_shape.join(" × ")}
          </p>
        </div>
        <div>
          <p class="text-gray-600">Min / Max</p>
          <p class="font-mono text-gray-900">
            {props.stats.min.toFixed(4)} / {props.stats.max.toFixed(4)}
          </p>
        </div>
        <div>
          <p class="text-gray-600">Mean / Std</p>
          <p class="font-mono text-gray-900">
            {props.stats.mean.toFixed(4)} / {props.stats.std.toFixed(4)}
          </p>
        </div>
      </div>
      <div class="mt-2">
        <p class="text-xs text-gray-500">
          Cube: {props.stats.cube_name} | Method: {props.method}
        </p>
      </div>
    </div>
  );
}

