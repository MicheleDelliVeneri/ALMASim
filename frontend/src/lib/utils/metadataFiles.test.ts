import { describe, expect, it } from 'vitest';
import {
	normalizeBackendPath,
	parseCsv,
	parseMetadataFile,
	rowsToCsv,
	snapshotMetadata
} from './metadataFiles';

describe('metadataFiles', () => {
	it('writes rows as CSV with the union of their columns, quoting when needed', () => {
		const csv = rowsToCsv([
			{ a: 1, b: 'x,y' },
			{ b: 'say "hi"', c: null }
		]);
		expect(csv).toBe('a,b,c\n1,"x,y",\n,"say ""hi""",\n');
		expect(rowsToCsv([])).toBe('');
	});

	it('parses CSV back, including quoted commas, quotes and CRLF', () => {
		expect(parseCsv('a,b\r\n1,"x,y"\r\n2,"say ""hi"""\n\n')).toEqual([
			{ a: '1', b: 'x,y' },
			{ a: '2', b: 'say "hi"' }
		]);
		expect(parseCsv('')).toEqual([]);
	});

	it('normalizes a backend path to a file with the chosen extension', () => {
		expect(normalizeBackendPath('results/run1.json', 'csv')).toBe('results/run1.csv');
		expect(normalizeBackendPath('results/', 'json')).toBe('results/metadata-results.json');
		expect(normalizeBackendPath('', 'json')).toBe('data.json');
	});

	it('reads JSON and CSV metadata files', async () => {
		const json = new File([JSON.stringify({ data: [{ a: 1 }] })], 'm.json');
		expect(await parseMetadataFile(json)).toEqual({ data: [{ a: 1 }], count: 1 });
		const csv = new File(['a\n1\n'], 'm.csv');
		expect(await parseMetadataFile(csv)).toEqual({ data: [{ a: '1' }], count: 1 });
		await expect(parseMetadataFile(new File(['{}'], 'bad.json'))).rejects.toThrow(
			'Invalid metadata file'
		);
	});

	it('snapshots a response with copied rows', () => {
		const response = { count: 1, data: [{ a: 1 }] };
		const snapshot = snapshotMetadata(response);
		expect(snapshot).toEqual(response);
		expect(snapshot.data[0]).not.toBe(response.data[0]);
	});
});
