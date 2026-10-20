<script lang="ts">
	import type { SimulationFormValues } from '#lib/utils/simulationForm.js';
	import type { MetadataRow } from '#lib/api/metadata.js';
	import {
		deriveArrayType,
		inferObservationConfigsFromMetadataRow
	} from '#lib/utils/observationPlan.js';
	import type { SimulationEstimate } from '#lib/api/simulation.js';

	interface Props {
		row: MetadataRow | null;
		getRowValue: (row: MetadataRow, key: string) => string;
		getRowNumber: (row: MetadataRow, key: string) => number | null;
		/** The simulation form's values (MADIV-28 T7.5). */
		form: SimulationFormValues;
		estimate: SimulationEstimate | null;
		estimating: boolean;
		estimateError: string | null;
	}

	let { row, getRowValue, getRowNumber, form, estimate, estimating, estimateError }: Props =
		$props();

	function derivePreviewSnr(targetRow: MetadataRow): string {
		if (!form.useMetadataSnr) return `${form.snr.toFixed(2)} (manual)`;
		const contSens =
			getRowNumber(targetRow, 'Cont_sens_mJybeam') ??
			getRowNumber(targetRow, 'Cont_sens') ??
			getRowNumber(targetRow, 'cont_sens');
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
	<section class="rounded-lg border border-accent-line bg-accent-soft p-6 shadow-panel">
		<h3 class="mb-4 text-lg font-semibold text-fg">Selected Row Preview</h3>
		<div class="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
			<div>
				<span class="font-medium text-fg-muted">Source:</span>
				<p class="mt-1 text-fg">
					{getRowValue(row, 'ALMA_source_name') || getRowValue(row, 'source_name')}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">RA:</span>
				<p class="mt-1 text-fg">
					{(() => {
						const val = getRowNumber(row, 'RA') ?? getRowNumber(row, 'ra');
						return val !== null ? val.toFixed(6) : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Imaging:</span>
				<p class="mt-1 text-fg">
					Mode {form.saveMode} | Lines {form.nLines} | Robust {form.robust.toFixed(2)}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Dec:</span>
				<p class="mt-1 text-fg">
					{(() => {
						const val = getRowNumber(row, 'Dec') ?? getRowNumber(row, 'dec');
						return val !== null ? val.toFixed(6) : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Band:</span>
				<p class="mt-1 text-fg">
					{getRowValue(row, 'Band') || getRowValue(row, 'band')}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">FOV:</span>
				<p class="mt-1 text-fg">
					{(() => {
						const val = getRowNumber(row, 'FOV') ?? getRowNumber(row, 'fov');
						return val !== null ? `${val.toFixed(2)} arcsec` : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Frequency:</span>
				<p class="mt-1 text-fg">
					{(() => {
						const val = getRowNumber(row, 'Freq') ?? getRowNumber(row, 'freq');
						return val !== null ? `${val.toFixed(2)} GHz` : 'N/A';
					})()}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">SNR:</span>
				<p class="mt-1 text-fg">{derivePreviewSnr(row)}</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">PWV:</span>
				<p class="mt-1 text-fg">
					{#if form.useMetadataPwv}
						{(() => {
							const val = getRowNumber(row, 'PWV') ?? getRowNumber(row, 'pwv');
							return val !== null ? `${val.toFixed(2)} mm (metadata)` : '1.00 mm (default)';
						})()}
					{:else}
						{`${form.pwvOverride.toFixed(2)} mm (manual)`}
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Array Type:</span>
				<p class="mt-1 text-fg">
					{getRowValue(row, 'Array_type') ||
						deriveArrayType(
							getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array')
						) ||
						'N/A'}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Inferred Configs:</span>
				<p class="mt-1 text-fg">
					{#if inferObservationConfigsFromMetadataRow(row, getRowNumber(row, 'Int.Time') ?? getRowNumber(row, 'int_time') ?? 3600)?.length}
						{inferObservationConfigsFromMetadataRow(
							row,
							getRowNumber(row, 'Int.Time') ?? getRowNumber(row, 'int_time') ?? 3600
						)
							?.map((config) => `${config.array_type}: ${config.antenna_array}`)
							.join(' | ')}
					{:else}
						{getRowValue(row, 'antenna_arrays') || getRowValue(row, 'antenna_array') || 'N/A'}
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Source Type:</span>
				<p class="mt-1 text-fg">{form.sourceType}</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Source Offset:</span>
				<p class="mt-1 text-fg">
					{form.sourceOffsetXArcsec.toFixed(1)}", {form.sourceOffsetYArcsec.toFixed(1)}"
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Imaging Algorithm:</span>
				<p class="mt-1 text-fg">{form.imagingAlgorithm}</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Cube Size:</span>
				<p class="mt-1 text-fg">
					{#if estimate}
						{estimate.n_pix} × {estimate.n_pix} × {estimate.n_channels}
					{:else if form.nPix !== null || form.nChannels !== null}
						{form.nPix ?? 'auto'} × {form.nPix ?? 'auto'} × {form.nChannels ?? 'auto'}
					{:else}
						Auto from metadata-derived FOV and frequency support
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Background:</span>
				<p class="mt-1 text-fg">
					{form.backgroundMode} @ level {form.backgroundLevel.toFixed(1)}
					{#if form.backgroundSeed !== null}
						(seed {form.backgroundSeed})
					{/if}
				</p>
			</div>
			<div>
				<span class="font-medium text-fg-muted">Estimated Size:</span>
				<p class="mt-1 text-fg">
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
				<span class="font-medium text-fg-muted">Estimated Outputs:</span>
				<p class="mt-1 text-fg">
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
