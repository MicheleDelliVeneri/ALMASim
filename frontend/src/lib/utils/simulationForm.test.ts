import { describe, expect, it } from 'vitest';
import { defaultSimulationForm, formRequestFields, parseOptionalInt } from './simulationForm';

describe('simulationForm', () => {
	it('parses optional integers, blank or invalid as null', () => {
		expect(parseOptionalInt(' 64 ')).toBe(64);
		expect(parseOptionalInt('')).toBeNull();
		expect(parseOptionalInt('abc')).toBeNull();
	});

	it('leaves metadata-driven and unset fields out of the request', () => {
		const fields = formRequestFields(defaultSimulationForm(''));
		expect(fields.snr).toBeUndefined();
		expect(fields.n_pix).toBeUndefined();
		expect(fields.n_lines).toBeUndefined();
		expect(fields.background_seed).toBeUndefined();
		expect(fields.output_dir).toBe('./outputs');
		expect(fields.source_type).toBe('point');
	});

	it('sends the manual overrides', () => {
		const form = {
			...defaultSimulationForm('/data'),
			useMetadataSnr: false,
			snr: 5,
			nPix: 128,
			nLines: 2,
			backgroundSeed: 7
		};
		const fields = formRequestFields(form);
		expect(fields).toMatchObject({
			snr: 5,
			n_pix: 128,
			n_lines: 2,
			background_seed: 7,
			output_dir: '/data'
		});
	});
});
