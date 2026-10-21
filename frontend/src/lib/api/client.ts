/** Errors from the console's API, shared by its client (`v2/client.ts`). */

/** FastAPI error bodies: `detail` is a string, or a list of validation errors (422). */
export interface ApiError {
	detail?: unknown;
}

/** An HTTP error from the console, with a readable message and the raw response body. */
export class ApiRequestError extends Error {
	constructor(
		message: string,
		readonly status: number,
		readonly body: unknown
	) {
		super(message);
		this.name = 'ApiRequestError';
	}
}

/** Turn a FastAPI `detail` into text: validation errors become "field: message; ...". */
export function formatErrorDetail(detail: unknown): string {
	if (typeof detail === 'string') return detail;
	if (Array.isArray(detail)) {
		return detail
			.map((item) => {
				if (item && typeof item === 'object' && 'msg' in item) {
					const loc = Array.isArray((item as { loc?: unknown }).loc)
						? (item as { loc: unknown[] }).loc.filter((part) => part !== 'body').join('.')
						: '';
					return loc
						? `${loc}: ${String((item as { msg: unknown }).msg)}`
						: String((item as { msg: unknown }).msg);
				}
				return typeof item === 'string' ? item : JSON.stringify(item);
			})
			.join('; ');
	}
	if (detail && typeof detail === 'object') {
		// The console's command checks: {"problems": ["Missing argument 'OUT'.", ...]}
		const problems = (detail as { problems?: unknown }).problems;
		if (Array.isArray(problems)) return problems.map(String).join('; ');
		return JSON.stringify(detail);
	}
	return '';
}
