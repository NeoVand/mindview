// A recorded run of the threads piece: what the reader and the painter computed for one prompt, read back from the GPU
// during a live run in a browser, kept so that the piece can play it back without the models (the landing's first
// run). The piece draws a recording exactly as it draws a live run; only the numbers come from a file. Numbers are
// stored as f16, the painter's pictures as WebP, and the painter's events keep the times they happened at.
//
// Files: recording.json (everything but the numbers and pictures), recording.bin (the numbers, f16, in the order
// listed in the JSON), and one .webp per picture.

/** What the reader's pass left, as the threads piece uses it (see Threads.gather). */
export interface ReaderTrace {
	tokens: string[]; // every token of the chat prompt
	W: number[]; // the prompt's words: their token indices
	NL: number; // layers read
	N: number; // row stride of probs (the reader's MAX_TOKENS)
	dims: { D: number; F: number; H: number; HD: number; KV: number };
	resid: Float32Array[][]; // [NL + 1][word]: the residual stream, D
	act: Float32Array[][]; // [NL][word]: the MLP's activations, F
	vals: Float32Array[]; // [NL]: attention values of every token, n * KV
	probs: Float32Array[][]; // [NL][head]: attention of each word to every token, nw * N
	e: [Float32Array, Float32Array]; // the two directions the threads are drawn along, D each
	proj: Float32Array; // [NL][F + Q][2]: every column of Wdown and Wo projected on them
	ghosts: { text: string; prob: number }[]; // [(NL + 1) * nw]: the logit lens's reading of each word at each layer
}

export type PaintEvent =
	| { t: number; kind: 'encoded'; v: number }
	| { t: number; kind: 'block'; s: number; b: number; attn: Float32Array; words: Float32Array }
	| { t: number; kind: 'stage'; s: number; res: number; image: Blob }
	| { t: number; kind: 'final'; image: Blob };

export interface Recording {
	prompt: string;
	seed: number;
	steps: number;
	fast: boolean;
	model: string; // which model file made it
	reader: ReaderTrace;
	paint: PaintEvent[]; // in order; t in seconds from the painting's start
}

// ---- f16

const f32 = new Float32Array(1),
	u32 = new Uint32Array(f32.buffer);

function toHalf(x: number) {
	f32[0] = x;
	const b = u32[0],
		s = (b >>> 16) & 0x8000,
		e = (b >>> 23) & 0xff,
		m = b & 0x7fffff;
	if (e === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
	const E = e - 127 + 15;
	if (E >= 0x1f) return s | 0x7c00;
	if (E <= 0) {
		if (E < -10) return s;
		const mm = (m | 0x800000) >> (1 - E);
		return s | ((mm + 0x1000) >> 13);
	}
	return s | (((E << 10) | (m >> 13)) + ((m >> 12) & 1));
}

function fromHalf(h: number) {
	const s = h & 0x8000 ? -1 : 1,
		e = (h >> 10) & 0x1f,
		m = h & 0x3ff;
	if (e === 0) return s * m * 2 ** -24;
	if (e === 0x1f) return m ? NaN : s * Infinity;
	return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---- writing

/** The recording as files: { name: contents }. */
export function pack(rec: Recording): Record<string, Blob | string> {
	const arrays: Float32Array[] = [];
	const add = (a: Float32Array) => (arrays.push(a), a.length);
	const r = rec.reader;
	r.resid.forEach((row) => row.forEach(add));
	r.act.forEach((row) => row.forEach(add));
	r.vals.forEach(add);
	r.probs.forEach((row) => row.forEach(add));
	r.e.forEach(add);
	add(r.proj);
	const files: Record<string, Blob | string> = {};
	const paint = rec.paint.map((e) => {
		if (e.kind === 'block') return { ...e, attn: add(e.attn), words: add(e.words) };
		if (e.kind === 'stage' || e.kind === 'final') {
			const name = e.kind === 'final' ? 'final.webp' : `pass${e.s + 1}-${e.res}.webp`;
			files[name] = e.image;
			return { ...e, image: name };
		}
		return e;
	});
	const total = arrays.reduce((a, x) => a + x.length, 0);
	const half = new Uint16Array(total);
	let o = 0;
	for (const a of arrays) for (let i = 0; i < a.length; i++) half[o++] = toHalf(a[i]);
	files['recording.bin'] = new Blob([half.buffer]);
	files['recording.json'] = JSON.stringify({
		version: 1,
		prompt: rec.prompt,
		seed: rec.seed,
		steps: rec.steps,
		fast: rec.fast,
		model: rec.model,
		reader: {
			tokens: r.tokens,
			W: r.W,
			NL: r.NL,
			N: r.N,
			dims: r.dims,
			n: r.vals[0].length / r.dims.KV,
			ghosts: r.ghosts
		},
		paint
	});
	return files;
}

// ---- reading

interface Stored {
	prompt: string;
	seed: number;
	steps: number;
	fast: boolean;
	model: string;
	reader: Pick<ReaderTrace, 'tokens' | 'W' | 'NL' | 'N' | 'dims' | 'ghosts'> & { n: number };
	paint: (
		| { t: number; kind: 'encoded'; v: number }
		| { t: number; kind: 'block'; s: number; b: number; attn: number; words: number }
		| { t: number; kind: 'stage'; s: number; res: number; image: string }
		| { t: number; kind: 'final'; image: string }
	)[];
}

/** A recording from its folder (see pack), pictures and all. */
export async function load(base: string): Promise<Recording> {
	const get = async (name: string) => {
		const r = await fetch(`${base}/${name}`);
		if (!r.ok) throw new Error(`Could not load the recording (${name}: ${r.status})`);
		return r;
	};
	const [meta, bin] = await Promise.all([
		get('recording.json').then((r) => r.json() as Promise<Stored>),
		get('recording.bin').then((r) => r.arrayBuffer())
	]);
	const half = new Uint16Array(bin);
	let o = 0;
	const take = (n: number) => {
		const a = new Float32Array(n);
		for (let i = 0; i < n; i++) a[i] = fromHalf(half[o++]);
		return a;
	};
	const m = meta.reader,
		{ D, F, H, HD, KV } = m.dims,
		nw = m.W.length,
		NL = m.NL;
	const resid = Array.from({ length: NL + 1 }, () => m.W.map(() => take(D)));
	const act = Array.from({ length: NL }, () => m.W.map(() => take(F)));
	const vals = Array.from({ length: NL }, () => take(m.n * KV));
	const probs = Array.from({ length: NL }, () => Array.from({ length: H }, () => take(nw * m.N)));
	const e: [Float32Array, Float32Array] = [take(D), take(D)];
	const proj = take(NL * (F + H * HD) * 2);
	const paint = await Promise.all(
		meta.paint.map(async (ev): Promise<PaintEvent> => {
			if (ev.kind === 'block') return { ...ev, attn: take(ev.attn), words: take(ev.words) };
			if (ev.kind === 'stage' || ev.kind === 'final')
				return {
					...ev,
					image: new Blob([await get(ev.image).then((r) => r.arrayBuffer())], {
						type: 'image/webp'
					})
				};
			return ev;
		})
	);
	if (o !== half.length) throw new Error('The recording is damaged (its numbers do not add up).');
	return {
		prompt: meta.prompt,
		seed: meta.seed,
		steps: meta.steps,
		fast: meta.fast,
		model: meta.model,
		reader: { ...m, resid, act, vals, probs, e, proj },
		paint
	};
}

/** A texture's pixels (rgba8unorm) as a WebP. */
export async function textureImage(device: GPUDevice, tex: GPUTexture): Promise<Blob> {
	const w = tex.width,
		h = tex.height,
		bpr = Math.ceil((w * 4) / 256) * 256;
	const buf = device.createBuffer({
		size: bpr * h,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});
	const enc = device.createCommandEncoder();
	enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
	device.queue.submit([enc.finish()]);
	await buf.mapAsync(GPUMapMode.READ);
	const src = new Uint8Array(buf.getMappedRange());
	const px = new Uint8ClampedArray(w * h * 4);
	for (let y = 0; y < h; y++) px.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
	buf.destroy();
	const c = new OffscreenCanvas(w, h);
	c.getContext('2d')!.putImageData(new ImageData(px, w, h), 0, 0);
	return c.convertToBlob({ type: 'image/webp', quality: 0.92 });
}
