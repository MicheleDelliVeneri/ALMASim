import { describe, expect, it } from 'vitest';

import {
	deriveArrayType,
	inferObservationConfigsFromMetadataRow,
	splitAntennaArrayByType
} from './observationPlan';

describe('deriveArrayType', () => {
	it.each([
		['DA41 DV02', '12m'],
		['CM01 CM02', '7m'],
		['PM01 PM02', 'TP'],
		['DA41 CM01', '12m+7m'],
		['DA41 PM01', '12m+TP'],
		['CM01 PM01', '7m+TP'],
		['DA41 CM01 PM01', '12m+7m+TP'],
		['da41 cm01', '12m+7m']
	])('classifies %s as %s', (antennas, expected) => {
		expect(deriveArrayType(antennas)).toBe(expected);
	});

	it('falls back to 12m when there are no recognised antennas', () => {
		expect(deriveArrayType('')).toBe('12m');
		expect(deriveArrayType('XX01')).toBe('12m');
	});
});

describe('splitAntennaArrayByType', () => {
	it('groups antennas by type in 12m, 7m, TP order', () => {
		expect(splitAntennaArrayByType('PM01 CM02 DA41  DV03 CM01')).toEqual([
			['12m', 'DA41 DV03'],
			['7m', 'CM02 CM01'],
			['TP', 'PM01']
		]);
	});

	it('returns no groups for an empty array', () => {
		expect(splitAntennaArrayByType('   ')).toEqual([]);
	});
});

describe('inferObservationConfigsFromMetadataRow', () => {
	it('builds one config per array type from antenna_arrays', () => {
		const configs = inferObservationConfigsFromMetadataRow(
			{ antenna_arrays: 'DA41 CM01' },
			3600,
			'BLC'
		);
		expect(configs).toEqual([
			{
				name: 'config_0_12m',
				array_type: '12m',
				antenna_array: 'DA41',
				total_time_s: 3600,
				correlator: 'BLC'
			},
			{
				name: 'config_1_7m',
				array_type: '7m',
				antenna_array: 'CM01',
				total_time_s: 3600,
				correlator: 'BLC'
			}
		]);
	});

	it('falls back to antenna_array and returns undefined when no antennas are listed', () => {
		expect(inferObservationConfigsFromMetadataRow({ antenna_array: 'PM02' }, 10)).toEqual([
			{
				name: 'config_0_TP',
				array_type: 'TP',
				antenna_array: 'PM02',
				total_time_s: 10,
				correlator: undefined
			}
		]);
		expect(inferObservationConfigsFromMetadataRow({}, 10)).toBeUndefined();
	});
});
