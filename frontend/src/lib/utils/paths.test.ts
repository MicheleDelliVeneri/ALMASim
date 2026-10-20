import { describe, expect, it } from 'vitest';

import { childPath } from './paths';

describe('childPath', () => {
	it('joins a folder name to its parent', () => {
		expect(childPath('/host_home/runs', ' sim-1 ')).toBe('/host_home/runs/sim-1');
	});

	it('does not double the separator under a trailing slash or the root', () => {
		expect(childPath('/data/', 'x')).toBe('/data/x');
		expect(childPath('/', 'x')).toBe('/x');
	});

	it('refuses names that are not a single folder', () => {
		for (const name of ['', '   ', '.', '..', 'a/b', 'a\\b']) {
			expect(childPath('/data', name)).toBeNull();
		}
	});
});
