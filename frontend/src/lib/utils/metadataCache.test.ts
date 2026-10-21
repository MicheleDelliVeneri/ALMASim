import { describe, expect, it } from 'vitest';

import {
	METADATA_CACHE_KEY,
	clearCachedResults,
	readCachedResults,
	writeCachedResults
} from './metadataCache';

class MemoryStorage implements Storage {
	private items = new Map<string, string>();
	get length() {
		return this.items.size;
	}
	clear() {
		this.items.clear();
	}
	getItem(key: string) {
		return this.items.get(key) ?? null;
	}
	key(index: number) {
		return [...this.items.keys()][index] ?? null;
	}
	removeItem(key: string) {
		this.items.delete(key);
	}
	setItem(key: string, value: string) {
		this.items.set(key, value);
	}
}

const row = { Project_abstract: 'a', QA2_status: 'Pass', Type: 'S' };

describe('metadataCache', () => {
	it('round-trips results', () => {
		const store = new MemoryStorage();
		writeCachedResults({ data: [row], count: 1 }, store);
		expect(readCachedResults(store)).toEqual({ data: [row], count: 1 });
	});

	it('drops a cache whose rows lack a required column', () => {
		const store = new MemoryStorage();
		store.setItem(METADATA_CACHE_KEY, JSON.stringify({ data: [{ Type: 'S' }], count: 1 }));
		expect(readCachedResults(store)).toBeNull();
		expect(store.getItem(METADATA_CACHE_KEY)).toBeNull();
	});

	it('returns null for missing or corrupt entries and without storage', () => {
		const store = new MemoryStorage();
		expect(readCachedResults(store)).toBeNull();
		store.setItem(METADATA_CACHE_KEY, '{not json');
		expect(readCachedResults(store)).toBeNull();
		expect(readCachedResults(null)).toBeNull();
	});

	it('reports results that do not fit and drops the older ones', () => {
		const store = new MemoryStorage();
		expect(writeCachedResults({ data: [row], count: 1 }, store)).toBe(true);
		const full = Object.assign(Object.create(store), {
			setItem: () => {
				throw new DOMException('quota', 'QuotaExceededError');
			}
		}) as Storage;
		expect(writeCachedResults({ data: [row, row], count: 2 }, full)).toBe(false);
		expect(readCachedResults(store)).toBeNull();
		expect(writeCachedResults({ data: [row], count: 1 }, null)).toBe(false);
	});

	it('clears the entry', () => {
		const store = new MemoryStorage();
		writeCachedResults({ data: [row], count: 1 }, store);
		clearCachedResults(store);
		expect(store.getItem(METADATA_CACHE_KEY)).toBeNull();
	});
});
