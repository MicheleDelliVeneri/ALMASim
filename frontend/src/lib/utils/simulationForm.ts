/**
 * The simulation form's values and how they become request fields
 * (MADIV-28 T7.5).
 */
import type { SimulationParamsCreate } from '#lib/api/simulation.js';

export interface SimulationFormValues {
	sourceType: string;
	nPix: number | null;
	nChannels: number | null;
	simulationName: string;
	outputDir: string;
	snr: number;
	useMetadataSnr: boolean;
	useMetadataPwv: boolean;
	pwvOverride: number;
	saveMode: string;
	msSaveMode: string;
	nLines: number;
	robust: number;
	imagingAlgorithm: 'legacy';
	numSimulations: number;
	sourceOffsetXArcsec: number;
	sourceOffsetYArcsec: number;
	backgroundMode: string;
	backgroundLevel: number;
	backgroundSeed: number | null;
}

export function defaultSimulationForm(outputDir: string): SimulationFormValues {
	return {
		sourceType: 'point',
		nPix: null,
		nChannels: null,
		simulationName: '',
		outputDir,
		snr: 1.3,
		useMetadataSnr: true,
		useMetadataPwv: true,
		pwvOverride: 1.0,
		saveMode: 'fits',
		msSaveMode: 'msv2',
		nLines: 0,
		robust: 0.0,
		imagingAlgorithm: 'legacy',
		numSimulations: 1,
		sourceOffsetXArcsec: 0.0,
		sourceOffsetYArcsec: 0.0,
		backgroundMode: 'none',
		backgroundLevel: 1.0,
		backgroundSeed: null
	};
}

/** An optional integer field: blank or unparseable is null. */
export function parseOptionalInt(text: string): number | null {
	const value = text.trim();
	if (value === '') return null;
	const parsed = parseInt(value, 10);
	return Number.isNaN(parsed) ? null : parsed;
}

/** The request fields the form sets directly (the rest come from the metadata row). */
export function formRequestFields(
	form: SimulationFormValues
): Pick<
	SimulationParamsCreate,
	| 'source_type'
	| 'n_pix'
	| 'n_channels'
	| 'snr'
	| 'save_mode'
	| 'ms_save_mode'
	| 'n_lines'
	| 'robust'
	| 'imaging_algorithm'
	| 'source_offset_x_arcsec'
	| 'source_offset_y_arcsec'
	| 'background_mode'
	| 'background_level'
	| 'background_seed'
	| 'output_dir'
> {
	return {
		source_type: form.sourceType,
		n_pix: form.nPix ?? undefined,
		n_channels: form.nChannels ?? undefined,
		snr: form.useMetadataSnr ? undefined : form.snr,
		save_mode: form.saveMode,
		ms_save_mode: form.msSaveMode,
		n_lines: form.nLines > 0 ? form.nLines : undefined,
		robust: form.robust,
		imaging_algorithm: form.imagingAlgorithm,
		source_offset_x_arcsec: form.sourceOffsetXArcsec,
		source_offset_y_arcsec: form.sourceOffsetYArcsec,
		background_mode: form.backgroundMode,
		background_level: form.backgroundLevel,
		background_seed: form.backgroundSeed ?? undefined,
		output_dir: form.outputDir || './outputs'
	};
}
