import { ApiRequestError } from '#lib/api/client.js';

/** One readable sentence for anything a request or action threw. */
export function errorMessage(error: unknown): string {
	if (error instanceof ApiRequestError) {
		if (error.status === 401) {
			return 'Not signed in to the console: open the link `almasim serve` printed.';
		}
		return error.message;
	}
	// fetch rejects with a TypeError when the server cannot be reached at all.
	if (error instanceof TypeError) return 'Cannot reach the ALMASim server.';
	if (error instanceof Error) return error.message || error.name;
	if (typeof error === 'string') return error;
	return 'Something went wrong.';
}
