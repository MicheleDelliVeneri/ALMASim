/**
 * The archive workflow's three runs (query, resolve, download), as command
 * lines; and what the page reads back from their CSV files.
 */
import { RUN_DIR } from './runDir';

export const RESULTS_CSV = 'results.csv';
export const PRODUCTS_CSV = 'products.csv';

/** The query: the form's filters, its results saved in the run's own folder. */
export function queryArgv(filters: string[]): string[] {
	return [...filters, '--save-csv', `${RUN_DIR}/${RESULTS_CSV}`, '--yes'];
}

/** Resolve the chosen observations' products into the run's folder. */
export function resolveArgv(memberOusUids: string[]): string[] {
	return [
		'products',
		'resolve',
		'--member-ous-uid',
		memberOusUids.join(','),
		'--save-products-csv',
		`${RUN_DIR}/${PRODUCTS_CSV}`
	];
}

export interface DownloadChoice {
	productsCsv: string;
	destination: string;
	productFilter: string;
	extract: boolean;
	unpack: boolean;
	calibrate: boolean;
}

/** Download the resolved products; unpacking and calibration follow on Slurm. */
export function downloadArgv(choice: DownloadChoice): string[] {
	const argv = [
		'products',
		'download',
		'--products-csv',
		choice.productsCsv,
		'--destination',
		choice.destination,
		'--product-filter',
		choice.productFilter
	];
	// Each step needs the one before it.
	if (choice.extract || choice.unpack || choice.calibrate) argv.push('--extract-tar');
	if (choice.unpack || choice.calibrate) argv.push('--unpack-ms');
	if (choice.calibrate) argv.push('--generate-calibrated-visibilities');
	return argv;
}

/** CSV rows as records keyed by column. */
export function toRecords(columns: string[], rows: string[][]): Record<string, string>[] {
	return rows.map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i] ?? ''])));
}

/** The distinct member OUS UIDs of the chosen rows, in order. */
export function memberOusUids(
	records: Record<string, string>[],
	chosen: Iterable<number>
): string[] {
	const seen = new Set<string>();
	for (const i of chosen) {
		const uid = records[i]?.member_ous_uid?.trim();
		if (uid) seen.add(uid);
	}
	return [...seen];
}

export interface ProductTotals {
	type: string;
	count: number;
	bytes: number;
}

/** Products per type (the download's --product-filter values), largest first. */
export function productTotals(records: Record<string, string>[]): ProductTotals[] {
	const byType = new Map<string, ProductTotals>();
	for (const record of records) {
		const type = record.product_type || 'other';
		const entry = byType.get(type) ?? { type, count: 0, bytes: 0 };
		entry.count += 1;
		entry.bytes += Number(record.content_length) || 0;
		byType.set(type, entry);
	}
	return [...byType.values()].sort((a, b) => b.bytes - a.bytes);
}

/** Bytes a filter downloads ("all", or one product type). */
export function filteredBytes(totals: ProductTotals[], filter: string): number {
	return totals
		.filter((t) => filter === 'all' || t.type === filter)
		.reduce((sum, t) => sum + t.bytes, 0);
}
