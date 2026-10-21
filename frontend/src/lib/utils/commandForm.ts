/**
 * A form for any command, from the console's catalogue (`/api/v2/commands`):
 * its values, the command line they make, the reverse for "edit and run",
 * and its sections (the CLI's help panels).
 */
import type { components } from '#lib/api/v2/schema.js';

export type Command = components['schemas']['Command'];
export type Param = components['schemas']['Param'];

/** A field's value: text (numbers too), on or off, or a list. */
export type FieldValue = string | boolean | string[];
export type Values = Record<string, FieldValue>;

/** Panels the form folds: the console fills them in, or few need them. */
export const FOLDED_PANELS = new Set(['Cluster', 'Advanced']);

export interface Section {
	title: string;
	params: Param[];
	folded: boolean;
}

/** The value a parameter starts with: its default. */
export function defaultValue(param: Param): FieldValue {
	if (param.type === 'boolean') return param.default === true;
	if (param.multiple) {
		return Array.isArray(param.default) ? param.default.map(String) : [];
	}
	return param.default == null ? '' : String(param.default);
}

export function initialValues(command: Command): Values {
	return Object.fromEntries(command.params.map((p) => [p.name, defaultValue(p)]));
}

function same(a: FieldValue, b: FieldValue): boolean {
	if (Array.isArray(a) && Array.isArray(b)) {
		return a.length === b.length && a.every((x, i) => x === b[i]);
	}
	return a === b;
}

/**
 * `almasim <argv>` for these values: the command's words, its arguments in
 * order, then each option that differs from its default. `extra` goes last
 * (the arguments `clean` passes to WSClean).
 */
export function buildArgv(command: Command, values: Values, extra: string[] = []): string[] {
	const argv = command.name.split(' ');
	for (const param of command.params.filter((p) => p.kind === 'argument')) {
		const value = values[param.name] ?? defaultValue(param);
		if (Array.isArray(value)) argv.push(...value.filter(Boolean));
		else if (typeof value === 'string' && value !== '') argv.push(value);
	}
	for (const param of command.params.filter((p) => p.kind === 'option')) {
		const value = values[param.name] ?? defaultValue(param);
		if (same(value, defaultValue(param))) continue;
		if (param.type === 'boolean') {
			if (value === true) argv.push(param.flags[0]);
			else if (param.negative_flags.length) argv.push(param.negative_flags[0]);
		} else if (Array.isArray(value)) {
			for (const item of value.filter(Boolean)) argv.push(param.flags[0], item);
		} else if (typeof value === 'string' && value !== '') {
			argv.push(param.flags[0], value);
		}
	}
	return [...argv, ...extra];
}

/**
 * The form for a command line, the reverse of `buildArgv`: what it sets,
 * and whatever the form has no field for (`extra`).
 */
export function parseArgv(command: Command, argv: string[]): { values: Values; extra: string[] } {
	const values = initialValues(command);
	const extra: string[] = [];
	const words = command.name.split(' ');
	let rest = argv.slice(0);
	if (words.every((w, i) => rest[i] === w)) rest = rest.slice(words.length);

	const byFlag = new Map<string, { param: Param; on: boolean }>();
	for (const param of command.params.filter((p) => p.kind === 'option')) {
		for (const flag of param.flags) byFlag.set(flag, { param, on: true });
		for (const flag of param.negative_flags) byFlag.set(flag, { param, on: false });
	}
	const positional = command.params.filter((p) => p.kind === 'argument');
	let nextArg = 0;
	const set = (param: Param, value: string) => {
		const current = values[param.name];
		if (param.multiple) values[param.name] = [...(Array.isArray(current) ? current : []), value];
		else values[param.name] = value;
	};
	// Multiple options start empty, so a parsed line does not add to the defaults.
	for (const param of command.params) if (param.multiple) values[param.name] = [];

	for (let i = 0; i < rest.length; i++) {
		const token = rest[i];
		if (token === '--') {
			extra.push(...rest.slice(i));
			break;
		}
		if (token.startsWith('-') && token.length > 1) {
			const [flag, inline] = token.includes('=') ? token.split(/=(.*)/s, 2) : [token, undefined];
			const match = byFlag.get(flag);
			if (!match) {
				extra.push(token);
				continue;
			}
			if (match.param.type === 'boolean') {
				values[match.param.name] = match.on;
			} else if (inline !== undefined) {
				set(match.param, inline);
			} else if (i + 1 < rest.length) {
				set(match.param, rest[++i]);
			}
			continue;
		}
		const param = positional[nextArg];
		if (!param) {
			extra.push(token);
		} else {
			set(param, token);
			if (!param.multiple) nextArg++;
		}
	}
	return { values, extra };
}

/** The command's inputs (its arguments) first, then one section per help panel. */
export function sections(command: Command): Section[] {
	const out: Section[] = [];
	const inputs = command.params.filter((p) => p.kind === 'argument');
	if (inputs.length) out.push({ title: 'Inputs', params: inputs, folded: false });
	const byPanel = new Map<string, Param[]>();
	for (const param of command.params.filter((p) => p.kind === 'option')) {
		const title = param.panel ?? 'Options';
		byPanel.set(title, [...(byPanel.get(title) ?? []), param]);
	}
	// Folded panels go last.
	const titles = [...byPanel.keys()].sort(
		(a, b) => Number(FOLDED_PANELS.has(a)) - Number(FOLDED_PANELS.has(b))
	);
	for (const title of titles) {
		const params = byPanel.get(title) ?? [];
		const folded = FOLDED_PANELS.has(title) && !params.some((p) => p.required);
		out.push({ title, params, folded });
	}
	return out;
}

/** Commands matching a search, over names and summaries. */
export function searchCommands(commands: Command[], query: string): Command[] {
	const words = query.toLowerCase().split(/\s+/).filter(Boolean);
	if (!words.length) return commands;
	return commands.filter((c) => {
		const text = `${c.name} ${c.summary}`.toLowerCase();
		return words.every((w) => text.includes(w));
	});
}
