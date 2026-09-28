// The Hugging Face Space (a static Space): the Paint page built on its own with plain Vite, into spaces/paint/dist.
//   pnpm space:paint   (then upload dist, see spaces/paint/README.md)
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
	root: here('.'),
	base: './',
	publicDir: here('public'),
	// the model always comes from Hugging Face here (src/lib/models.ts)
	define: { __LOCAL_MODELS__: 'false', __LOCAL_PACKED__: 'false' },
	plugins: [
		tailwindcss(),
		svelte({
			configFile: false,
			compilerOptions: {
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			}
		})
	],
	resolve: { alias: { $lib: here('../../src/lib'), '$app/paths': here('app-paths.ts') } },
	build: { outDir: here('dist'), emptyOutDir: true, target: 'es2022' }
});
