// $app/paths for the Space's plain Vite build (src/lib/models.ts uses asset() only for local model files, which the
// Space never has): everything is served from the Space's root.
export const base = '';
export const assets = '';
export const asset = (path: string) => path;
export const resolve = (path: string) => path;
