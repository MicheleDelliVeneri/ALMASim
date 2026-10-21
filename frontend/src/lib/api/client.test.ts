import { describe, expect, it } from 'vitest';

import { formatErrorDetail } from './client';

describe('formatErrorDetail', () => {
	it('keeps string details', () => {
		expect(formatErrorDetail('Download job not found')).toBe('Download job not found');
	});

	it('turns FastAPI validation errors into readable text', () => {
		const detail = [
			{ loc: ['body', 'destination'], msg: 'Field required', type: 'missing' },
			{ loc: ['query', 'limit'], msg: 'Input should be a valid integer', type: 'int_parsing' }
		];
		expect(formatErrorDetail(detail)).toBe(
			'destination: Field required; query.limit: Input should be a valid integer'
		);
	});

	it("lists the console's command problems", () => {
		expect(
			formatErrorDetail({ problems: ["Missing argument 'OUT'.", 'No such option: --x'] })
		).toBe("Missing argument 'OUT'.; No such option: --x");
	});

	it('falls back to JSON for other shapes and to "" for nothing', () => {
		expect(formatErrorDetail({ code: 7 })).toBe('{"code":7}');
		expect(formatErrorDetail(undefined)).toBe('');
	});
});
