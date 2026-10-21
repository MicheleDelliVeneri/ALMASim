/**
 * The last metadata query results, kept in localStorage so the metadata,
 * simulations and data pages share them across navigation.
 */
import type { MetadataResponse } from '../api/metadata';

export const METADATA_CACHE_KEY = 'almasim:metadata-results';

/** Columns every cached row must have; a cache from before they existed is dropped. */
const REQUIRED_COLUMNS = ['Project_abstract', 'QA2_status', 'Type'];

function browserStorage(): Storage | null {
	try {
		return typeof window === 'undefined' ? null : window.localStorage;
	} catch {
		return null;
	}
}

export function readCachedResults(
	store: Storage | null = browserStorage()
): MetadataResponse | null {
	if (!store) return null;
	try {
		const raw = store.getItem(METADATA_CACHE_KEY);
		if (!raw) return null;
		const parsed = JSON.parse(raw) as MetadataResponse;
		const sample = parsed?.data?.[0];
		if (sample && REQUIRED_COLUMNS.some((col) => !(col in sample))) {
			store.removeItem(METADATA_CACHE_KEY);
			return null;
		}
		return parsed;
	} catch {
		return null;
	}
}

/**
 * Cache `data` for the other pages; whether it fitted. When it does not
 * (localStorage holds a few MB), the previous results are removed too, so no
 * page goes on showing rows from an older query.
 */
export function writeCachedResults(
	data: MetadataResponse,
	store: Storage | null = browserStorage()
): boolean {
	if (!store) return false;
	try {
		store.setItem(METADATA_CACHE_KEY, JSON.stringify(data));
		return true;
	} catch {
		clearCachedResults(store);
		return false;
	}
}

/** Shown when results are too large to hand to the other pages. */
export const CACHE_FULL_MESSAGE =
	'These results are too large to keep in the browser, so the Simulations page cannot use them. Save them to a CSV file and load that instead.';

export function clearCachedResults(store: Storage | null = browserStorage()): void {
	try {
		store?.removeItem(METADATA_CACHE_KEY);
	} catch {
		/* storage blocked */
	}
}
