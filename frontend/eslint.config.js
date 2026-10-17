import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';

export default ts.config(
	{ ignores: ['build/', '.svelte-kit/', 'dist/'] },
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs.recommended,
	prettier,
	...svelte.configs.prettier,
	{
		languageOptions: {
			globals: { ...globals.browser, ...globals.node }
		}
	},
	{
		files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
		languageOptions: {
			parserOptions: {
				parser: ts.parser,
				extraFileExtensions: ['.svelte']
			}
		}
	},
	{
		// New in eslint-plugin-svelte 3's recommended set; the code predates them.
		rules: {
			// Sets here are copied and reassigned (`s = new Set(s)`), which is already reactive.
			'svelte/prefer-svelte-reactivity': 'off',
			'svelte/require-each-key': 'warn',
			// No `paths.base` is configured, so plain hrefs resolve correctly.
			'svelte/no-navigation-without-resolve': 'warn'
		}
	}
);
