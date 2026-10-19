import { defineConfig } from 'vitest/config';

// Unit tests for framework-free modules (pure TypeScript under src/lib).
export default defineConfig({
	test: {
		include: ['src/**/*.test.ts'],
		environment: 'node'
	}
});
