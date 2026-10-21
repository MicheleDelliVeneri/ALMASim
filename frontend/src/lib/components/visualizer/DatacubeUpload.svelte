<script lang="ts">
	interface Props {
		onFileUpload: (event: Event) => void;
		integrationMethod: 'sum' | 'mean';
		complexComponent: 'real' | 'imag' | 'magnitude' | 'phase';
		channelStart: string;
		channelEnd: string;
		onMethodChange: (method: 'sum' | 'mean') => void;
		onComplexComponentChange: (component: 'real' | 'imag' | 'magnitude' | 'phase') => void;
		onChannelStartChange: (value: string) => void;
		onChannelEndChange: (value: string) => void;
		loading: boolean;
	}

	let {
		onFileUpload,
		integrationMethod,
		complexComponent,
		channelStart,
		channelEnd,
		onMethodChange,
		onComplexComponentChange,
		onChannelStartChange,
		onChannelEndChange,
		loading
	}: Props = $props();
</script>

<div class="space-y-4">
	<!-- Divider -->
	<div class="relative">
		<div class="absolute inset-0 flex items-center">
			<div class="w-full border-t border-line"></div>
		</div>
		<div class="relative flex justify-center text-sm">
			<span class="bg-surface px-2 text-fg-dim">OR</span>
		</div>
	</div>

	<!-- File Upload -->
	<div class="space-y-2">
		<label for="datacube-upload" class="block text-sm font-medium text-fg-muted"
			>Upload Product (.npz or FITS file)</label
		>
		<div class="flex items-center space-x-4">
			<input
				id="datacube-upload"
				type="file"
				accept=".npz,.fits,.fit,.fts"
				onchange={onFileUpload}
				disabled={loading}
				class="block w-full text-sm text-fg-dim file:mr-4 file:rounded-md file:border-0 file:bg-accent-soft file:px-4 file:py-2 file:text-sm file:font-semibold file:text-accent hover:file:bg-accent-soft disabled:opacity-50"
			/>
		</div>
	</div>

	<!-- Integration Method -->
	<div class="space-y-2">
		<label for="integration-method" class="block text-sm font-medium text-fg-muted"
			>Integration Method</label
		>
		<select
			id="integration-method"
			value={integrationMethod}
			onchange={(e) => onMethodChange(e.currentTarget.value as 'sum' | 'mean')}
			disabled={loading}
			class="rounded-md border border-line px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
		>
			<option value="sum">Sum (integrate all channels)</option>
			<option value="mean">Mean (average over channels)</option>
		</select>
	</div>

	<div class="space-y-2">
		<label for="complex-component" class="block text-sm font-medium text-fg-muted"
			>Complex Component</label
		>
		<select
			id="complex-component"
			value={complexComponent}
			onchange={(e) =>
				onComplexComponentChange(e.currentTarget.value as 'real' | 'imag' | 'magnitude' | 'phase')}
			disabled={loading}
			class="rounded-md border border-line px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
		>
			<option value="real">Real</option>
			<option value="imag">Imaginary</option>
			<option value="magnitude">Magnitude</option>
			<option value="phase">Phase</option>
		</select>
	</div>

	<div class="space-y-2">
		<span class="block text-sm font-medium text-fg-muted">Channel Range (inclusive)</span>
		<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
			<input
				type="number"
				min="0"
				value={channelStart}
				oninput={(e) => onChannelStartChange(e.currentTarget.value)}
				disabled={loading}
				placeholder="Start channel"
				class="rounded-md border border-line px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
			/>
			<input
				type="number"
				min="0"
				value={channelEnd}
				oninput={(e) => onChannelEndChange(e.currentTarget.value)}
				disabled={loading}
				placeholder="End channel"
				class="rounded-md border border-line px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
			/>
		</div>
		<p class="text-xs text-fg-dim">Leave blank to integrate all channels on axis 0.</p>
	</div>
</div>
