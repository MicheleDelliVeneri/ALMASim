<script lang="ts">
	interface Props {
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

	let { stats, method }: Props = $props();
</script>

<div class="space-y-2 rounded-md bg-gray-50 p-4">
	<h3 class="text-sm font-semibold text-gray-900">Datacube Statistics</h3>
	<div class="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
		<div>
			<p class="text-gray-600">Original Shape</p>
			<p class="font-mono text-gray-900">
				{stats.shape.join(' × ')}
			</p>
		</div>
		<div>
			<p class="text-gray-600">Integrated Shape</p>
			<p class="font-mono text-gray-900">
				{stats.integrated_shape.join(' × ')}
			</p>
		</div>
		<div>
			<p class="text-gray-600">Min / Max</p>
			<p class="font-mono text-gray-900">
				{stats.min.toFixed(4)} / {stats.max.toFixed(4)}
			</p>
		</div>
		<div>
			<p class="text-gray-600">Mean / Std</p>
			<p class="font-mono text-gray-900">
				{stats.mean.toFixed(4)} / {stats.std.toFixed(4)}
			</p>
		</div>
	</div>
	<div class="mt-2">
		<p class="text-xs text-gray-500">
			Cube: {stats.cube_name} | Method: {method}
		</p>
	</div>
</div>
