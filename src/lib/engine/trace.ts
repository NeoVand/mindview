// Loads a generation trace exported by research/scripts/export_trace.py.

export type LensWord = [string, number];

export interface Manifest {
	prompt: string;
	encoder: string;
	size: number;
	steps: number;
	seed: number;
	arrays: Record<string, { file: string; dtype: 'float16'; shape: number[] }>;
	act1: {
		tokens: string[];
		layers: number;
		hidden: number;
		taps: number[];
		heads: number;
		lens: LensWord[][][]; // [layer][token][top5]
	};
	decoder?: { file: string; res: number; label: string }[];
	act3: {
		blocks: number;
		double_blocks: number;
		grid: number;
		sigmas: number[];
		n_text_tokens: number;
		text_len: number;
	};
}

export interface Tensor {
	data: Float32Array;
	shape: number[];
}

export interface Trace {
	base: string;
	manifest: Manifest;
	arrays: Record<string, Tensor>;
	lens: ImageBitmap[]; // [step * blocks + block], TAEF2 previews, 256²
	x0: ImageBitmap[]; // per step, full VAE
	final: ImageBitmap;
	decoder: { bitmap: ImageBitmap; res: number; label: string }[]; // VAE decoder stages, coarse to fine
}

function halfToFloat(src: Uint16Array): Float32Array {
	const out = new Float32Array(src.length);
	for (let i = 0; i < src.length; i++) {
		const h = src[i];
		const s = h & 0x8000 ? -1 : 1;
		const e = (h >> 10) & 0x1f;
		const f = h & 0x3ff;
		if (e === 0) out[i] = s * f * 2 ** -24;
		else if (e === 31) out[i] = f ? NaN : s * Infinity;
		else out[i] = s * (1 + f / 1024) * 2 ** (e - 15);
	}
	return out;
}

async function bitmap(url: string): Promise<ImageBitmap> {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
	return createImageBitmap(await res.blob());
}

export async function loadTrace(
	base: string,
	onProgress?: (fraction: number) => void
): Promise<Trace> {
	const res = await fetch(`${base}/manifest.json`);
	if (!res.ok) throw new Error(`Could not load the trace manifest at ${base} (${res.status})`);
	const manifest: Manifest = await res.json();
	const { steps } = manifest;
	const blocks = manifest.act3.blocks;
	const arrayKeys = Object.keys(manifest.arrays);
	const total = arrayKeys.length + steps * blocks + steps + 1;
	let done = 0;
	const tick = () => onProgress?.(++done / total);

	const arrays: Record<string, Tensor> = {};
	await Promise.all(
		arrayKeys.map(async (key) => {
			const meta = manifest.arrays[key];
			const r = await fetch(`${base}/${meta.file}`);
			if (!r.ok) throw new Error(`Could not load ${meta.file} (${r.status})`);
			arrays[key] = {
				data: halfToFloat(new Uint16Array(await r.arrayBuffer())),
				shape: meta.shape
			};
			tick();
		})
	);

	const lensUrls: string[] = [];
	for (let s = 0; s < steps; s++)
		for (let b = 0; b < blocks; b++)
			lensUrls.push(`${base}/lens_s${s}_b${String(b).padStart(2, '0')}.png`);
	const lens = await Promise.all(lensUrls.map((u) => bitmap(u).then((b) => (tick(), b))));
	const x0 = await Promise.all(
		Array.from({ length: steps }, (_, s) => bitmap(`${base}/x0_s${s}.png`).then((b) => (tick(), b)))
	);
	const final = await bitmap(`${base}/final.png`);
	tick();
	// decoder stages are stored at their native resolution; upscale without smoothing so their grid stays visible
	const decoder = await Promise.all(
		(manifest.decoder ?? []).map(async (d) => {
			const b = await bitmap(`${base}/${d.file}`);
			const size = final.width;
			const up =
				b.width === size
					? b
					: await createImageBitmap(b, {
							resizeWidth: size,
							resizeHeight: size,
							resizeQuality: 'pixelated'
						});
			return { bitmap: up, res: d.res, label: d.label };
		})
	);
	return { base, manifest, arrays, lens, x0, final, decoder };
}
