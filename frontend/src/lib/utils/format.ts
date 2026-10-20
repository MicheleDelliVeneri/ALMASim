/** Display formatting shared by the pages (MADIV-28 T7.3). */

const UNITS = ['KB', 'MB', 'GB', 'TB'] as const;

/**
 * A byte count with binary (1024) steps: whole bytes, one decimal for KB and
 * MB, two for GB and above.
 */
export function formatBytes(bytes: number): string {
	if (Math.abs(bytes) < 1024) return `${Math.round(bytes)} B`;
	let value = bytes;
	let unit: (typeof UNITS)[number] = 'KB';
	for (unit of UNITS) {
		value /= 1024;
		if (Math.abs(value) < 1024 || unit === 'TB') break;
	}
	return `${value.toFixed(unit === 'KB' || unit === 'MB' ? 1 : 2)} ${unit}`;
}

/** A transfer rate, formatted like {@link formatBytes}. */
export function formatSpeed(bytesPerSecond: number): string {
	return `${formatBytes(bytesPerSecond)}/s`;
}

/**
 * A date in the browser's locale. Numbers are Unix times in seconds; text that
 * is not a date is returned unchanged.
 */
export function formatDate(value: string | number): string {
	const date = new Date(typeof value === 'number' ? value * 1000 : value);
	return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
