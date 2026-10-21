/**
 * Path of a new folder `name` inside `parent`, or null when `name` is not a
 * single folder name (empty, ".", ".." or containing a slash).
 */
export function childPath(parent: string, name: string): string | null {
	const trimmed = name.trim();
	if (!trimmed || trimmed === '.' || trimmed === '..' || /[/\\]/.test(trimmed)) return null;
	return `${parent.replace(/\/+$/, '')}/${trimmed}`;
}

/**
 * A server path for a `{path:path}` URL segment: each part percent-encoded so
 * `#`, `?`, `%` and spaces reach the server, the slashes kept.
 */
export function encodePathSegments(path: string): string {
	return path.split('/').map(encodeURIComponent).join('/');
}
