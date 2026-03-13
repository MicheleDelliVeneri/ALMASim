import pino, { type Logger, type LoggerOptions } from 'pino/browser';

const level = import.meta.env.VITE_LOG_LEVEL ?? (import.meta.env.DEV ? 'debug' : 'info');

const LEVEL_STYLES: Record<string, string> = {
	trace: 'color:#aaa;font-weight:bold',
	debug: 'color:#9b59b6;font-weight:bold',
	info: 'color:#2196f3;font-weight:bold',
	warn: 'color:#f39c12;font-weight:bold',
	error: 'color:#e74c3c;font-weight:bold',
	fatal: 'color:#c0392b;font-weight:bold'
};

function getConsoleFn(levelName: string): typeof console.log {
	if (levelName === 'warn') return console.warn;
	if (levelName === 'error' || levelName === 'fatal') return console.error;
	return console.log;
}

function makeWriter(levelName: string) {
	const style = LEVEL_STYLES[levelName] ?? 'font-weight:bold';
	const consoleFn = getConsoleFn(levelName);

	return (obj: Record<string, unknown>) => {
		const { msg, scope, level: _l, time: _t, ...rest } = obj;
		const tag = levelName.toUpperCase().padEnd(5);
		const scopeStr = typeof scope === 'string' ? scope : '';
		const msgStr = typeof msg === 'string' ? msg : '';
		const prefix = scopeStr ? `[${scopeStr}] ` : '';
		const hasExtra = Object.keys(rest).length > 0;

		if (hasExtra) {
			consoleFn(`%c${tag}%c ${prefix}${msgStr}`, style, 'color:inherit', rest);
		} else {
			consoleFn(`%c${tag}%c ${prefix}${msgStr}`, style, 'color:inherit');
		}
	};
}

const loggerOptions: LoggerOptions = {
	level,
	browser: {
		asObject: true,
		write: {
			trace: makeWriter('trace'),
			debug: makeWriter('debug'),
			info: makeWriter('info'),
			warn: makeWriter('warn'),
			error: makeWriter('error'),
			fatal: makeWriter('fatal')
		}
	}
};

const rootLogger = pino(loggerOptions);

export function createLogger(scope: string): Logger {
	return rootLogger.child({ scope });
}

export const logger = rootLogger;
