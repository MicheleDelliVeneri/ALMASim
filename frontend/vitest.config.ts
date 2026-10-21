import { sveltekit } from '@sveltejs/kit/vite';
import { svelteTesting } from '@testing-library/svelte/vite';
import { defineConfig } from 'vitest/config';

// Two kinds of test: framework-free modules under src/lib (`*.test.ts`, in
// Node), and components and rune stores (`*.svelte.test.ts`, in jsdom with
// Testing Library).
export default defineConfig({
	plugins: [sveltekit()],
	test: {
		projects: [
			{
				extends: true,
				test: {
					name: 'unit',
					include: ['src/**/*.test.ts'],
					exclude: ['src/**/*.svelte.test.ts'],
					environment: 'node'
				}
			},
			{
				extends: true,
				plugins: [svelteTesting()],
				test: {
					name: 'component',
					include: ['src/**/*.svelte.test.ts'],
					environment: 'jsdom',
					setupFiles: ['./vitest-setup.ts']
				}
			}
		]
	}
});
