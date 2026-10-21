import { describe, expect, it } from 'vitest';

import {
	downloadArgv,
	filteredBytes,
	memberOusUids,
	productTotals,
	queryArgv,
	resolveArgv,
	toRecords
} from './archive';
import { inRunDir } from './runDir';

describe('the archive runs', () => {
	it('save their CSV files in the run folder', () => {
		expect(queryArgv(['metadata', '--band', '3'])).toEqual([
			'metadata', '--band', '3', '--save-csv', '{run_dir}/results.csv', '--yes'
		]); // prettier-ignore
		expect(resolveArgv(['uid://A001/X1/X1', 'uid://A001/X1/X2'])).toEqual([
			'products', 'resolve',
			'--member-ous-uid', 'uid://A001/X1/X1,uid://A001/X1/X2',
			'--save-products-csv', '{run_dir}/products.csv'
		]); // prettier-ignore
		expect(inRunDir('{run_dir}/results.csv', '/runs/01J')).toBe('/runs/01J/results.csv');
	});

	it('download with each post-processing step the next one needs', () => {
		const base = {
			productsCsv: '/runs/R/products.csv',
			destination: '/data/archive',
			productFilter: 'raw',
			extract: false,
			unpack: false,
			calibrate: false
		};
		expect(downloadArgv(base)).toEqual([
			'products', 'download', '--products-csv', '/runs/R/products.csv',
			'--destination', '/data/archive', '--product-filter', 'raw'
		]); // prettier-ignore
		expect(downloadArgv({ ...base, calibrate: true }).slice(8)).toEqual([
			'--extract-tar',
			'--unpack-ms',
			'--generate-calibrated-visibilities'
		]);
		expect(downloadArgv({ ...base, unpack: true }).slice(8)).toEqual([
			'--extract-tar',
			'--unpack-ms'
		]);
	});
});

describe('reading the CSV files back', () => {
	const records = toRecords(
		['ALMA_source_name', 'member_ous_uid'],
		[['A', 'uid://1'], ['B', 'uid://2'], ['B2', 'uid://2'], ['C', ''], ['D']]
	);

	it('keys rows by column, short rows padded', () => {
		expect(records[4]).toEqual({ ALMA_source_name: 'D', member_ous_uid: '' });
	});

	it('gives each chosen observation once', () => {
		expect(memberOusUids(records, [0, 1, 2, 3, 4])).toEqual(['uid://1', 'uid://2']);
		expect(memberOusUids(records, new Set([2]))).toEqual(['uid://2']);
	});

	it('totals the products per type, largest first', () => {
		const products = toRecords(
			['product_type', 'content_length'],
			[
				['raw', '1000'],
				['cubes', '5000'],
				['raw', '2500'],
				['', 'x']
			]
		);
		const totals = productTotals(products);
		expect(totals).toEqual([
			{ type: 'cubes', count: 1, bytes: 5000 },
			{ type: 'raw', count: 2, bytes: 3500 },
			{ type: 'other', count: 1, bytes: 0 }
		]);
		expect(filteredBytes(totals, 'all')).toBe(8500);
		expect(filteredBytes(totals, 'raw')).toBe(3500);
	});
});
