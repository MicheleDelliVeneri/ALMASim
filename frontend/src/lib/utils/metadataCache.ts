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

export function writeCachedResults(
	data: MetadataResponse,
	store: Storage | null = browserStorage()
): void {
	try {
		store?.setItem(METADATA_CACHE_KEY, JSON.stringify(data));
	} catch {
		/* storage full or blocked: the cache is a convenience */
	}
}

export function clearCachedResults(store: Storage | null = browserStorage()): void {
	try {
		store?.removeItem(METADATA_CACHE_KEY);
	} catch {
		/* storage blocked */
	}
}
