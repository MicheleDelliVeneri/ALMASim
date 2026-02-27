<script lang="ts">
	interface Props {
		row: Record<string, unknown> | null;
		getRowValue: (row: Record<string, unknown>, key: string) => string;
		getRowNumber: (row: Record<string, unknown>, key: string) => number | null;
		sourceType: string;
		nPix: number;
		nChannels: number;
		snr: number;
		saveMode: string;
		nLines: number;
		robust: number;
	}

	let { row, getRowValue, getRowNumber, sourceType, nPix, nChannels, snr, saveMode, nLines, robust }: Props = $props();
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
				<span class="font-medium text-gray-700">Source Type:</span>
				<p class="mt-1 text-gray-900">{sourceType}</p>
			</div>
			<div>
				<span class="font-medium text-gray-700">Cube Size:</span>
				<p class="mt-1 text-gray-900">
					{nPix} × {nPix} × {nChannels}
				</p>
			</div>
		</div>
	</section>
{/if}
