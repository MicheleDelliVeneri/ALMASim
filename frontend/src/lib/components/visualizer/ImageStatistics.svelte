<script lang="ts">
	interface Props {
		stats: {
			shape: number[];
			window_shape?: number[];
			integrated_shape: number[];
			min: number;
			max: number;
			mean: number;
			std: number;
			cube_name: string;
			format?: string;
			integration_axis?: number;
			complex_component?: string;
			channel_start?: number;
			channel_end?: number;
		};
		method: string;
	}

	let { stats, method }: Props = $props();
</script>

<div class="space-y-2 rounded-md bg-gray-50 p-4">
	<h3 class="text-sm font-semibold text-gray-900">Product Statistics</h3>
	<div class="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
		<div>
			<p class="text-gray-600">Original Shape</p>
			<p class="font-mono text-gray-900">
				{stats.shape.join(' × ')}
			</p>
		</div>
		<div>
			<p class="text-gray-600">Integrated Window</p>
			<p class="font-mono text-gray-900">
				{(stats.window_shape ?? stats.integrated_shape).join(' × ')}
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
			Product: {stats.cube_name} | Format: {(stats.format ?? 'array').toUpperCase()} | Component: {stats.complex_component ?? 'n/a'} | Method: {method} | Axis: {stats.integration_axis ?? 'n/a'} | Channels: {stats.channel_start ?? 'n/a'}–{stats.channel_end ?? 'n/a'}
		</p>
	</div>
</div>
