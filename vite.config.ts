import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { existsSync } from 'node:fs';

// The models: from static/models when they are there (development), otherwise from Hugging Face (see src/lib/models.ts).
// MODELS=hub or MODELS=local decides it outright.
const localModels = process.env.MODELS
	? process.env.MODELS === 'local'
	: existsSync('static/models/ternary-bonsai-1.7b/model.gguf') &&
		existsSync('static/models/bonsai-image-4b/manifest.json');
// Served from a folder (GitHub Pages serves a project's site at /<repository>): BASE_PATH=/mindview
const base = (process.env.BASE_PATH ?? '') as '' | `/${string}`;

export default defineConfig({
	define: { __LOCAL_MODELS__: JSON.stringify(localModels) },
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// every page is prerendered as a shell (the pieces run in the browser); unknown paths get the app too
			adapter: adapter({ fallback: '404.html' }),
			paths: { base }
		})
	],
	test: {
		expect: { requireAssertions: true },
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					browser: {
						enabled: true,
						provider: playwright(),
						instances: [{ browser: 'chromium', headless: true }]
					},
					include: ['src/**/*.svelte.{test,spec}.{js,ts}'],
					exclude: ['src/lib/server/**']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'server',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
});
