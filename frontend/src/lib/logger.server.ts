import pino, { type Logger } from 'pino';

const level = process.env.VITE_LOG_LEVEL ?? (process.env.NODE_ENV === 'production' ? 'info' : 'debug');

const transport = pino.transport({
	target: 'pino-pretty',
	options: {
		colorize: true,
		translateTime: 'SYS:HH:MM:ss',
		ignore: 'pid,hostname',
		messageFormat: '{scope} {msg}'
	}
});

const rootLogger = pino({ level }, transport);

export function createLogger(scope: string): Logger {
	return rootLogger.child({ scope });
}

export const logger = rootLogger;
