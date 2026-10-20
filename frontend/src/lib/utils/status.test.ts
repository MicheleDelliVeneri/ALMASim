import { describe, expect, it } from 'vitest';

import { isActiveStatus, isTerminalStatus, statusTone } from './status';

describe('status', () => {
	it('treats waiting and working jobs as active', () => {
		for (const s of ['queued', 'pending', 'starting', 'running', 'downloading']) {
			expect(isActiveStatus(s)).toBe(true);
			expect(isTerminalStatus(s)).toBe(false);
		}
	});

	it('treats finished jobs as terminal', () => {
		for (const s of ['completed', 'failed', 'cancelled']) {
			expect(isTerminalStatus(s)).toBe(true);
			expect(isActiveStatus(s)).toBe(false);
		}
	});

	it('gives each status a tone', () => {
		expect(statusTone('queued')).toBe('queued');
		expect(statusTone('downloading')).toBe('active');
		expect(statusTone('completed')).toBe('ok');
		expect(statusTone('failed')).toBe('danger');
		expect(statusTone('cancelled')).toBe('neutral');
		expect(statusTone('something-new')).toBe('neutral');
	});
});
