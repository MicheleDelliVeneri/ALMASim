<script lang="ts">
	import { deriveArrayType, inferObservationConfigsFromMetadataRow } from '$lib/utils/observationPlan';
	import type { SimulationEstimate } from '$lib/api/simulation';

	interface Props {
		row: Record<string, unknown> | null;
		getRowValue: (row: Record<string, unknown>, key: string) => string;
		getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
		sourceType: string;
		nPix: number | null;
		nChannels: number | null;
		snr: number;
		useMetadataSnr: boolean;
		useMetadataPwv: boolean;
		pwvOverride: number;
		saveMode: string;
		nLines: number;
		robust: number;
		estimate: SimulationEstimate | null;
		estimating: boolean;
		estimateError: string | null;
	}

	let { row, getRowValue, getRowNumber, sourceType, nPix, nChannels, snr, useMetadataSnr, useMetadataPwv, pwvOverride, saveMode, nLines, robust, estimate, estimating, estimateError }: Props = $props();

	function derivePreviewSnr(targetRow: Record<string, unknown>): string {
		if (!useMetadataSnr) return `${snr.toFixed(2)} (manual)`;
		const contSens = getRowNumber(targetRow, 'Cont_sens_mJybeam') ?? getRowNumber(targetRow, 'Cont_sens') ?? getRowNumber(targetRow, 'cont_sens');
		const lineSens = getRowNumber(targetRow, 'Line_sens_10kms_mJybeam');
		if (contSens !== null && lineSens !== null && contSens > 0) {
			const inferred = Math.max(1.0, Math.min(30.0, lineSens / contSens));
			return `${inferred.toFixed(2)} (metadata heuristic)`;
		}
		if (contSens !== null) return 'Auto from metadata sensitivity';
		return '1.30 (fallback)';
	}
</script>

{#if row}
	<section class="rounded-lg border border-blue-200 bg-blue-50 p-6 shadow-md">
		<h3 class="mb-4 text-lg font-semibold text-gray-900">Selected Row Preview</h3>
		<div class="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
			<div>
				<span class="font-medium text-gray-700">Source:</span>
				<p class="mt-1 text-gray-900">
					{getRowValue(row, 'ALMA_source_name') || getRowValue(row, 'source_name')}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">RA:</span>
				<p class="mt-1 text-gray-900">
					{(() => {
						const val = getRowNumber(row, 'RA') ?? getRowNumber(row, 'ra');
						return val !== null ? val.toFixed(6) : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Dec:</span>
				<p class="mt-1 text-gray-900">
					{(() => {
						const val = getRowNumber(row, 'Dec') ?? getRowNumber(row, 'dec');
						return val !== null ? val.toFixed(6) : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Band:</span>
				<p class="mt-1 text-gray-900">
					{getRowValue(row, 'Band') || getRowValue(row, 'band')}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">FOV:</span>
				<p class="mt-1 text-gray-900">
					{(() => {
						const val = getRowNumber(row, 'FOV') ?? getRowNumber(row, 'fov');
						return val !== null ? `${val.toFixed(2)} arcsec` : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Frequency:</span>
				<p class="mt-1 text-gray-900">
					{(() => {
						const val = getRowNumber(row, 'Freq') ?? getRowNumber(row, 'freq');
						return val !== null ? `${val.toFixed(2)} GHz` : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">SNR:</span>
				<p class="mt-1 text-gray-900">{derivePreviewSnr(row)}</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">PWV:</span>
				<p class="mt-1 text-gray-900">
					{#if useMetadataPwv}
						{(() => {
							const val = getRowNumber(row, 'PWV') ?? getRowNumber(row, 'pwv');
							return val !== null ? `${val.toFixed(2)} mm (metadata)` : '1.00 mm (default)';
						})()}
					{:else}
						{`${pwvOverride.toFixed(2)} mm (manual)`}
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Array Type:</span>
				<p class="mt-1 text-gray-900">
					{getRowValue(row, 'Array_type') || deriveArrayType(getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array')) || 'N/A'}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Inferred Configs:</span>
				<p class="mt-1 text-gray-900">
					{#if inferObservationConfigsFromMetadataRow(row, getRowNumber(row, 'Int.Time') ?? getRowNumber(row, 'int_time') ?? 3600)?.length}
						{inferObservationConfigsFromMetadataRow(row, getRowNumber(row, 'Int.Time') ?? getRowNumber(row, 'int_time') ?? 3600)
							?.map((config) => `${config.array_type}: ${config.antenna_array}`)
							.join(' | ')}
					{:else}
						{getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array') || 'N/A'}
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Source Type:</span>
				<p class="mt-1 text-gray-900">{sourceType}</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Cube Size:</span>
				<p class="mt-1 text-gray-900">
					{#if estimate}
						{estimate.n_pix} × {estimate.n_pix} × {estimate.n_channels}
					{:else if nPix !== null || nChannels !== null}
						{nPix ?? 'auto'} × {nPix ?? 'auto'} × {nChannels ?? 'auto'}
					{:else}
						Auto from metadata-derived FOV and frequency support
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Estimated Size:</span>
				<p class="mt-1 text-gray-900">
					{#if estimating}
						Computing…
					{:else if estimate}
						{estimate.raw_single_cube_gb.toFixed(3)} GiB per float32 cube
					{:else if estimateError}
						Unavailable
					{:else}
						Select a row to estimate
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Estimated Outputs:</span>
				<p class="mt-1 text-gray-900">
					{#if estimate}
						~{estimate.estimated_standard_output_gb.toFixed(3)} GiB raw
					{:else if estimateError}
						{estimateError}
					{:else}
						Waiting for estimate
					{/if}
				</p>
			</div>
		</div>
	</section>
{/if}
