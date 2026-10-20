/**
 * Columns, filters and the virtual window of the metadata results table
 * (MADIV-28 T7.5).
 */

/** Always first, always visible, not reorderable. */
export const PINNED_COLUMNS = ['proposal_id', 'ALMA_source_name'];

/** Every known column's label, in the preferred display order. */
export const COLUMN_LABELS: Record<string, string> = {
	ALMA_source_name: 'Source Name',
	Band: 'Band',
	Array_type: 'Array Type',
	antenna_arrays: 'Antenna Arrays',
	'Ang.res.': 'Ang. Res. (arcsec)',
	'Obs.date': 'Obs. Date',
	Project_abstract: 'Project Abstract',
	science_keyword: 'Science Keyword',
	scientific_category: 'Science Category',
	QA2_status: 'QA2 Status',
	Type: 'Project Type',
	PWV: 'PWV',
	SB_name: 'SB Name',
	'Vel.res.': 'Vel. Res. (km/s)',
	RA: 'R.A. (deg)',
	Dec: 'Dec (deg)',
	FOV: 'FOV (arcsec)',
	'Int.Time': 'Int. Time (s)',
	Cont_sens_mJybeam: 'Cont. Sens. (mJy/beam)',
	Line_sens_10kms_mJybeam: 'Line Sens. 10km/s (mJy/beam)',
	Bandwidth: 'Bandwidth (GHz)',
	Freq: 'Frequency (GHz)',
	'Freq.sup.': 'Freq. Support',
	proposal_id: 'Proposal ID',
	member_ous_uid: 'Member OUS UID',
	group_ous_uid: 'Group OUS UID'
};

const PREFERRED_ORDER = [
	'Band',
	'Array_type',
	'Ang.res.',
	'Obs.date',
	'Project_abstract',
	'science_keyword',
	'scientific_category',
	'QA2_status',
	'Type'
];

/** The reorderable columns' starting order: the preferred ones, then the rest. */
export const DEFAULT_COLUMN_ORDER = [
	...PREFERRED_ORDER,
	...Object.keys(COLUMN_LABELS).filter(
		(c) => !PREFERRED_ORDER.includes(c) && !PINNED_COLUMNS.includes(c)
	)
];

/** Hidden until asked for: verbose or rarely needed. */
export const DEFAULT_HIDDEN = [
	'antenna_arrays',
	'PWV',
	'SB_name',
	'Vel.res.',
	'Cont_sens_mJybeam',
	'Line_sens_10kms_mJybeam',
	'Freq.sup.',
	'group_ous_uid'
];

export function formatColumnName(column: string): string {
	if (COLUMN_LABELS[column]) return COLUMN_LABELS[column];
	return column
		.split('_')
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(' ');
}

export function stringifyValue(value: unknown): string {
	if (Array.isArray(value)) return value.join(', ');
	if (typeof value === 'object' && value) return JSON.stringify(value);
	return value?.toString() ?? '';
}

/**
 * Case-insensitive match of a cell against a filter: a substring, or with
 * `*` a glob over the whole cell ("2023*.L" matches what starts with 2023
 * and ends with .L).
 */
export function matchesFilter(cellValue: string, pattern: string): boolean {
	const cell = cellValue.toLowerCase();
	const pat = pattern.toLowerCase();
	if (!pat.includes('*')) return cell.includes(pat);
	const escaped = pat.replace(/([.+?^${}()|[\]\\])/g, '\\$1').replace(/\*/g, '.*');
	try {
		return new RegExp(`^${escaped}$`).test(cell);
	} catch {
		return cell.includes(pat);
	}
}

/** Positions in `rows` of the rows every non-empty filter matches. */
export function filterRowIndices(
	rows: readonly Record<string, unknown>[],
	filters: Record<string, string>
): number[] {
	const active = Object.entries(filters).filter(([, v]) => v.length > 0);
	const indices: number[] = [];
	rows.forEach((row, index) => {
		if (active.every(([col, val]) => matchesFilter(String(row[col] ?? ''), val))) {
			indices.push(index);
		}
	});
	return indices;
}

/** `order` with `column` moved to where `target` is; unchanged if either is missing. */
export function moveColumn(order: readonly string[], column: string, target: string): string[] {
	const next = [...order];
	const from = next.indexOf(column);
	const to = next.indexOf(target);
	if (from === -1 || to === -1 || from === to) return next;
	next.splice(from, 1);
	next.splice(to, 0, column);
	return next;
}

export interface VirtualWindow {
	start: number;
	end: number;
	topPad: number;
	bottomPad: number;
}

/** The rows to render for a scroll position, with `overscan` rows either side. */
export function virtualWindow(
	scrollTop: number,
	totalRows: number,
	rowHeight: number,
	viewportHeight: number,
	overscan: number
): VirtualWindow {
	const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
	const end = Math.min(totalRows, Math.ceil((scrollTop + viewportHeight) / rowHeight) + overscan);
	return {
		start,
		end,
		topPad: start * rowHeight,
		bottomPad: Math.max(0, (totalRows - end) * rowHeight)
	};
}
