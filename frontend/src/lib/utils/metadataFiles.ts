/**
 * Reading and writing metadata result files (JSON or CSV) for the metadata page
 * (MADIV-28 T7.5: moved out of routes/metadata/+page.svelte).
 */
import type { MetadataResponse } from '../api/metadata';
import { formatDateStamp } from '../utils';

export const DEFAULT_METADATA_PATH = 'data';

export const snapshotMetadata = (data: MetadataResponse): MetadataResponse => ({
	count: typeof data.count === 'number' ? data.count : data.data.length,
	data: data.data.map((row) => ({ ...row }))
});

export const rowsToCsv = (rows: Record<string, unknown>[]): string => {
	if (!rows.length) return '';
	const columnsSet = new Set<string>();
	const columns: string[] = [];
	for (const row of rows) {
		for (const key of Object.keys(row)) {
			if (!columnsSet.has(key)) {
				columnsSet.add(key);
				columns.push(key);
			}
		}
	}
	const escape = (val: unknown): string => {
		if (val === null || val === undefined) return '';
		const s = typeof val === 'string' ? val : JSON.stringify(val);
		return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
	};
	const header = columns.map(escape).join(',');
	const body = rows.map((r) => columns.map((c) => escape(r[c])).join(',')).join('\n');
	return `${header}\n${body}\n`;
};

export const downloadMetadata = (
	data: MetadataResponse,
	path: string,
	format: 'json' | 'csv' = 'json'
) => {
	if (typeof window === 'undefined') return;
	const snapshot = snapshotMetadata(data);
	const extension = format === 'csv' ? '.csv' : '.json';
	const body = format === 'csv' ? rowsToCsv(snapshot.data) : JSON.stringify(snapshot, null, 2);
	const mime = format === 'csv' ? 'text/csv' : 'application/json';
	const blob = new Blob([body], { type: mime });
	const safePath = path.replace(/^[.@/\\]+/, '');
	const sanitized =
		safePath.length > 0 ? safePath.replace(/\s+/g, '-').replace(/[^\w./-]/g, '_') : '';
	const segments = sanitized.split('/').filter(Boolean);
	let fileName = segments.length ? segments.join('_') : `metadata-results-${formatDateStamp()}`;
	fileName = fileName.replace(/\.(json|csv)$/i, '');
	fileName += extension;
	const link = document.createElement('a');
	link.href = URL.createObjectURL(blob);
	link.download = fileName;
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
	URL.revokeObjectURL(link.href);
};

export const normalizeBackendPath = (input: string, format: 'json' | 'csv' = 'json') => {
	const extension = format === 'csv' ? '.csv' : '.json';
	const cleaned = input.replace(/^[./\\]+/, '').trim() || DEFAULT_METADATA_PATH;
	// Strip any existing json/csv extension so the format selector wins.
	const withoutExt = cleaned.replace(/\.(json|csv)$/i, '');
	// If the user provided no leaf filename (just a directory), append the default name.
	const hasFileLeaf = /[^/]+$/.test(withoutExt) && !withoutExt.endsWith('/');
	const base = hasFileLeaf ? withoutExt : `${withoutExt.replace(/\/$/, '')}/metadata-results`;
	return `${base}${extension}`;
};

export const parseCsv = (input: string): Array<Record<string, string>> => {
	const rows: string[][] = [];
	let current = '';
	let inQuotes = false;
	const currentRow: string[] = [];

	const pushValue = () => {
		currentRow.push(current);
		current = '';
	};

	const pushRow = () => {
		pushValue();
		rows.push([...currentRow]);
		currentRow.length = 0;
	};

	for (let i = 0; i < input.length; i++) {
		const char = input[i];
		if (char === '"') {
			if (inQuotes && input[i + 1] === '"') {
				current += '"';
				i++;
			} else {
				inQuotes = !inQuotes;
			}
		} else if (char === ',' && !inQuotes) {
			pushValue();
		} else if ((char === '\n' || char === '\r') && !inQuotes) {
			if (char === '\r' && input[i + 1] === '\n') i++;
			pushRow();
		} else {
			current += char;
		}
	}

	if (current.length > 0 || currentRow.length > 0) {
		pushRow();
	}

	if (!rows.length) return [];
	const [header, ...dataRows] = rows;
	return dataRows
		.filter((cells) => cells.some((cell) => cell.trim().length))
		.map((cells) =>
			header.reduce<Record<string, string>>((acc, column, index) => {
				acc[column] = cells[index] ?? '';
				return acc;
			}, {})
		);
};

export const parseMetadataFile = async (file: File): Promise<MetadataResponse> => {
	const text = await file.text();
	const extension = file.name.split('.').pop()?.toLowerCase();
	if (extension === 'csv') {
		const data = parseCsv(text);
		return { data, count: data.length };
	}

	try {
		const parsed = JSON.parse(text) as Partial<MetadataResponse>;
		if (!parsed || !Array.isArray(parsed.data)) {
			throw new Error();
		}
		return {
			data: parsed.data,
			count:
				typeof parsed.count === 'number'
					? parsed.count
					: Array.isArray(parsed.data)
						? parsed.data.length
						: 0
		};
	} catch {
		throw new Error('Invalid metadata file. Provide JSON with a `data` array or a CSV file.');
	}
};
