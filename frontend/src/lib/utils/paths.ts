/**
 * Path of a new folder `name` inside `parent`, or null when `name` is not a
 * single folder name (empty, ".", ".." or containing a slash).
 */
export function childPath(parent: string, name: string): string | null {
	const trimmed = name.trim();
	if (!trimmed || trimmed === '.' || trimmed === '..' || /[/\\]/.test(trimmed)) return null;
	return `${parent.replace(/\/+$/, '')}/${trimmed}`;
}
