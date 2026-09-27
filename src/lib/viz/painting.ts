// The painter's part of the journey, computed in order through the scheduler, laid out as a panel beside the path
// with one row per kind of thing, a column per block, read from left to right in the order it happens:
//   pictures  each block's picture (what its output has in mind, through the decoder's first stage: 64 x 64); after a
//             pass's last block, its guess of the finished picture goes through the rest of the decoder, each stage
//             larger as it doubles: 128 x 128, 256 x 256, 512 x 512
//   readings  under each picture, the block's reading: where the picture's 1,024 patches read each of your words, in
//             the words' colours (the attention inside the block, which comes before its picture)
//   words     the words' threads, continuing from the reader as a cable beneath the readings: at every block each
//             reaches up towards the place where the picture reads it most, touching the reading when it is read
//             hardest; arcs between the threads show how the words read each other in that block (see
//             word-weave.ts). While the decoder works they run on in the cable, unread
//   braid     before all that, in the gap after the reader: each word's states after layers 7, 14 and 21 merge into
//             one strand (the adapter joins the three), which runs on as the word's thread
//   finale    the finished picture, with each word's reading averaged over every block of every pass beside it
// Units: a 'tick' is one step along the painter: tick = pass * 29 + block + 1 for the 25 blocks, then pass * 29 + 26,
// 27, 28 for the decoder's stages (and 29 is the next pass's start).
import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import type { Painter } from '$lib/runtime/painter';
import type { GpuScheduler } from '$lib/runtime/scheduler';
import { Gallery, READING } from './gallery';
import { LineLayer } from './lines';
import type { ImagePlanes } from './planes';
import { WordWeave, type Stitch } from './word-weave';

type V3 = [number, number, number];
export const GRID = 32;
export const TICKS_PER_STEP = 29;

// the panel (y) and its columns (x)
const BLOCK = 2.6; // side of a block's picture and reading
const STAGE: Record<number, number> = { 128: 3.4, 256: 4.3, 512: 5.6 }; // sides of the decoder's stages
const GAP = 0.3;
const PICTURES = 1.2; // bottom edge of the pictures row
const READINGS = PICTURES - BLOCK - 0.5; // bottom edge of the readings row
const CABLE: V3 = [0, READINGS - 1.1, 0.9]; // the words' cable, beneath the readings and a little in front
const FINAL = 10; // side of the finished picture

/** Width of each tick's column within a pass: blocks, then the decoder's stages. */
const WIDTHS = Array.from({ length: TICKS_PER_STEP }, (_, r) =>
	r === 0 ? GAP : r <= 25 ? BLOCK + GAP : r <= 28 ? STAGE[64 << (r - 25)] + GAP * 2 : GAP
);
const CENTRES: number[] = []; // centre of tick r's column, from the start of its pass
{
	let e = 0;
	for (let r = 0; r < TICKS_PER_STEP; r++) {
		CENTRES.push(e + WIDTHS[r] / 2);
		e += WIDTHS[r];
	}
	CENTRES.push(e + GAP); // the next pass's start
}
const PASS = CENTRES[TICKS_PER_STEP];

export interface PaintingLayout {
	x0: number; // where the painter begins along the journey
	readerEnd: number; // where the reader's last layer stands
	layers: number; // the reader's layer count
	taps: number[]; // the reader's layers the painter listens to
}

/** A picture of side `side` facing the side the camera rides on: centred on x, its bottom edge at y. */
const card = (x: number, bottom: number, side: number): [V3, V3, V3] => [
	[x - side / 2, bottom + side, 0],
	[side, 0, 0],
	[0, -side, 0]
];

/** How bright the past is at `age` ticks behind the cut (the gallery dims the same way). */
const fade = (age: number, tau = 30, floor = 0.4) => Math.max(floor, Math.exp(-age / tau));

export class Painting {
	readonly braid: LineLayer;
	/** Readouts received, in ticks. */
	ticks = 0;
	/** How much of the text side (re-reading + adapter) has run, 0..1. */
	encoded = 0;
	done = false;
	readonly steps: number;
	readonly x0: number;
	private gallery: Gallery;
	private weave: WordWeave;
	private canvases: { plane: number; tick: number; side: number }[] = [];
	private finalPlane = -1;
	private textures: GPUTexture[] = [];
	private maps: GPUTexture[] = []; // the averaged maps shown beside the finished picture
	private mapPlanes: number[] = [];
	private sum: Float32Array; // attention summed over all blocks so far, for the maps at the end
	private reads = 0;
	private finale: { spots: { y: number; z: number }[]; side: number };
	private doneAt = 0;
	private words: { first: number; count: number };

	constructor(
		private device: GPUDevice,
		frame: GPUBuffer,
		private planes: ImagePlanes,
		private layout: PaintingLayout,
		private palette: Float32Array, // word colours (linear rgb) by word index
		readonly wordCount: number,
		steps: number,
		/** Fast: the one-file model's 2 passes, the first a sketch at 256 x 256 (16 x 16 patches). */
		readonly fast = false
	) {
		this.steps = steps;
		this.x0 = layout.x0;
		this.braid = new LineLayer(device, frame, true);
		this.words = { first: 3, count: wordCount };
		const n = wordCount;
		this.finale = beside(FINAL, n);
		this.sum = new Float32Array(GRID * GRID * n);
		this.gallery = new Gallery(device, frame, steps * 25 * 2 + 8);
		// each word's place around the cable, in the order of the words
		const r = Math.min(0.75, 0.25 + 0.03 * n);
		this.weave = new WordWeave(
			device,
			frame,
			palette,
			n,
			steps,
			TICKS_PER_STEP,
			(t) => card(this.xOf(t), READINGS, BLOCK),
			(t, k) => {
				const a = (2 * Math.PI * k) / n;
				return [this.xOf(t), CABLE[1] + r * Math.cos(a), CABLE[2] + r * Math.sin(a)];
			}
		);
		for (let k = 0; k < n; k++) {
			const tex = device.createTexture({
				size: [GRID, GRID],
				format: 'rgba8unorm',
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
			});
			this.maps.push(tex);
			this.textures.push(tex);
		}
	}

	get ticksTotal() {
		return this.steps * TICKS_PER_STEP;
	}

	/** World length of the painting, from its start to the end of its last pass. */
	get length() {
		return this.steps * PASS;
	}

	/** Where tick t stands along the journey (columns are as wide as what stands in them). */
	xOf(tick: number) {
		if (tick <= 0) return this.x0 + tick * GAP;
		const s = Math.min(this.steps - 1, Math.floor(tick / TICKS_PER_STEP)),
			r = tick - s * TICKS_PER_STEP;
		const i = Math.min(TICKS_PER_STEP - 1, Math.floor(r)),
			f = r - i;
		return this.x0 + s * PASS + CENTRES[i] + (CENTRES[i + 1] - CENTRES[i]) * f;
	}

	/** Centre of the finished picture. */
	get finalAt(): V3 {
		// far enough past the last pass that the maps on its left stand clear of it
		return [this.x0 + this.length + FINAL + 5, (READINGS + PICTURES + BLOCK) / 2, 0];
	}

	/**
	 * Queue the whole painting, in order: re-read the prompt (512 tokens), adapter, the passes block by block, the
	 * final decode. taps: each word's position in the reader after layers 7, 14 and 21.
	 */
	start(
		painter: Painter,
		llm: BonsaiLLM,
		scheduler: GpuScheduler,
		prompt: string,
		taps: V3[][],
		seed: number
	) {
		painter.setSteps(this.steps);
		if (this.fast) painter.textLength = 256;
		painter.setNoise(undefined, seed, this.fast);
		this.buildBraid(taps);
		this.mapPlanes = this.maps.map((t) => this.planes.add(t));
		const { tasks } = painter.encodeTasks(llm, prompt, scheduler.slice);
		tasks.forEach((t, i) => {
			const before = t.done;
			t.done = async () => {
				await before?.();
				this.encoded = (i + 1) / tasks.length;
			};
		});
		scheduler.push(...tasks);
		const early = painter.taef2.earlyTexture;
		for (let s = 0; s < this.steps; s++)
			scheduler.push(
				...painter.stepTasks(
					s,
					this.words,
					(b, attn, _space, words) => this.read(s, b, attn, words),
					{
						// each stage of the decoder is copied as it is made, and stands as its work finishes
						record: (enc, res) =>
							this.keep(
								res === 512 ? painter.taef2.texture : painter.taef2.stageTextures.get(res)!,
								s * TICKS_PER_STEP + 25 + Math.log2(res / 64),
								STAGE[res],
								enc
							),
						done: (res) => {
							this.ticks = Math.max(this.ticks, s * TICKS_PER_STEP + 26 + Math.log2(res / 64));
						}
					},
					scheduler.slice,
					// every block's picture goes into the gallery as soon as it is made
					(enc, b) => {
						const tick = s * TICKS_PER_STEP + b + 1;
						this.gallery.addPicture(enc, early, tick, ...card(this.xOf(tick), PICTURES, BLOCK));
					}
				)
			);
		scheduler.push(
			...painter.finalTasks(() => {
				this.finalPlane = this.keep(painter.taef2.texture, -1, 0);
				// beside the finished picture: each word's reading over the whole painting
				this.paintFinaleMaps(this.sum.map((x) => x / Math.max(1, this.reads)));
				this.done = true;
				this.doneAt = performance.now();
				this.ticks = this.ticksTotal;
			}, scheduler.slice)
		);
	}

	/**
	 * The adapter, in the gap between the reader and the painter. For each word: arcs from its beads after layers 7 and
	 * 14 and its thread's own end after layer 21 meet at one point (the adapter joins the three into one vector);
	 * from there one strand runs on to where the word's thread enters the painter. A strand appears as the 1.7B's second reading
	 * passes its layer (w = share of the text side done).
	 */
	private buildBraid(taps: V3[][]) {
		const L = this.layout,
			gap = this.x0 - L.readerEnd;
		const merge = L.readerEnd + gap * 0.4;
		const pts: number[] = [],
			attr: number[] = [];
		const smooth = (t: number) => {
			const u = Math.max(0, Math.min(1, t));
			return u * u * (3 - 2 * u);
		};
		const readShare = 0.9; // of the text side's work, the 1.7B's re-reading (the adapter is the rest)
		let strand = 0;
		taps.forEach((t, k) => {
			const end = t[2];
			const M: V3 = [merge, end[1], end[2]];
			t.forEach((p, j) => {
				const w0 = readShare * (L.taps[j] / L.layers),
					w1 = j === 2 ? readShare : readShare * 0.98;
				// an arc out of the bundle and over the later layers (a straight run for the thread's own end)
				const my = (p[1] + M[1]) / 2,
					mz = (p[2] + M[2]) / 2,
					mr = Math.hypot(my, mz) || 1;
				const bulge = j === 2 ? 0 : 0.12 * (M[0] - p[0]);
				const K = j === 2 ? 12 : 64;
				for (let q = 0; q <= K; q++) {
					const u = q / K,
						b = Math.sin(Math.PI * u) * bulge;
					pts.push(
						p[0] + (M[0] - p[0]) * u,
						p[1] + (M[1] - p[1]) * u + (my / mr) * b,
						p[2] + (M[2] - p[2]) * u + (mz / mr) * b,
						w0 + (w1 - w0) * u
					);
					attr.push(strand | (k << 16) | ((j === 2 ? 220 : 120) << 24));
				}
				strand++;
			});
			// joined: on to where the word's thread enters the painter
			const to = this.weave.entrances[k];
			const K = 48;
			for (let q = 0; q <= K; q++) {
				const u = q / K,
					e = smooth(u);
				pts.push(
					M[0] + (to[0] - M[0]) * u,
					M[1] + (to[1] - M[1]) * e,
					M[2] + (to[2] - M[2]) * e,
					readShare + (1 - readShare) * u
				);
				attr.push(strand | (k << 16) | (230 << 24));
			}
			strand++;
		});
		this.braid.set(new Float32Array(pts), new Uint32Array(attr), this.palette);
	}

	/**
	 * One block's reading: its card in the readings row, where each word's thread comes down at the place it is read
	 * most; and the words' reading of each other (words: [word][word]).
	 */
	private read(s: number, b: number, a: Float32Array, words?: Float32Array) {
		// a sketch pass reads 16 x 16 patches: each stands for 2 x 2 of the full grid (shown as the blocks they are)
		const attn = a.length === GRID * GRID * this.wordCount ? a : enlarge(a, this.wordCount);
		for (let i = 0; i < attn.length; i++) this.sum[i] += attn[i];
		this.reads++;
		const tick = s * TICKS_PER_STEP + b + 1;
		const { maps, gain, share } = this.mapsOf(attn);
		this.weave.add(
			s,
			b,
			maps.map((m, j) => ({ ...peak(m), share: share[j] })),
			words
		);
		const n = this.wordCount,
			P = GRID * GRID;
		// the reading: every word's map in its colour, added up
		const light = new Float32Array(P * 3);
		let top = 1e-9;
		for (let j = 0; j < n; j++) {
			const c = this.palette.subarray(j * 4, j * 4 + 3),
				g = gain[j];
			for (let i = 0; i < P; i++) {
				const v = maps[j][i] ** 2 * g; // squared: where a word is read most stands out
				light[i * 3] += c[0] * v;
				light[i * 3 + 1] += c[1] * v;
				light[i * 3 + 2] += c[2] * v;
			}
		}
		for (let i = 0; i < P; i++)
			top = Math.max(
				top,
				0.2126 * light[i * 3] + 0.7152 * light[i * 3 + 1] + 0.0722 * light[i * 3 + 2]
			);
		const px = new Uint8Array(READING * READING * 4),
			enc = (x: number) => Math.round(255 * Math.pow(Math.min(1, Math.max(0, x)), 1 / 2.2));
		for (let i = 0; i < P; i++) {
			px[i * 4] = enc(light[i * 3] / top);
			px[i * 4 + 1] = enc(light[i * 3 + 1] / top);
			px[i * 4 + 2] = enc(light[i * 3 + 2] / top);
			px[i * 4 + 3] = 255;
		}
		// the reading happens inside the block, just before its picture: it appears a moment earlier
		this.gallery.addReading(px, tick - 0.3, ...card(this.xOf(tick), READINGS, BLOCK), 1);
		this.ticks = Math.max(this.ticks, tick + 1);
	}

	/**
	 * Every word's map from attention [patch][word]: 0..1, bright where a patch reads the word more than the average
	 * patch does (scaled to the word's strongest patch); and each word's brightness, lower when the picture reads the
	 * word less than it reads the most-read word.
	 */
	private mapsOf(attn: Float32Array) {
		const n = this.wordCount,
			P = GRID * GRID;
		const mass = new Float32Array(n),
			peak = new Float32Array(n);
		for (let i = 0; i < P; i++)
			for (let j = 0; j < n; j++) {
				const a = attn[i * n + j];
				mass[j] += a;
				if (a > peak[j]) peak[j] = a;
			}
		const top = Math.max(1e-12, ...mass);
		const maps = Array.from({ length: n }, (_, j) => {
			const mean = mass[j] / P,
				k = 1 / Math.max(1e-12, peak[j] - mean);
			const v = new Float32Array(P);
			for (let i = 0; i < P; i++) v[i] = Math.min(1, Math.max(0, (attn[i * n + j] - mean) * k));
			return v;
		});
		const share = Array.from(mass, (m) => m / top);
		const gain = share.map((x) => 1.5 * (0.3 + 0.7 * Math.sqrt(x)));
		return { maps, gain, share };
	}

	/** The maps beside the finished picture (in the word's colour, display sRGB for the picture planes). */
	private paintFinaleMaps(attn: Float32Array) {
		const { maps, gain } = this.mapsOf(attn);
		const px = new Uint8Array(GRID * GRID * 4),
			enc = (x: number) => Math.round(255 * Math.pow(Math.min(1, Math.max(0, x)), 1 / 2.2));
		maps.forEach((v, j) => {
			const c = this.palette.subarray(j * 4, j * 4 + 3),
				g = gain[j] / 1.5;
			for (let i = 0; i < v.length; i++) {
				px[i * 4] = enc(c[0] * v[i] * g);
				px[i * 4 + 1] = enc(c[1] * v[i] * g);
				px[i * 4 + 2] = enc(c[2] * v[i] * g);
				px[i * 4 + 3] = 255;
			}
			this.device.queue.writeTexture({ texture: this.maps[j] }, px, { bytesPerRow: GRID * 4 }, [
				GRID,
				GRID
			]);
		});
	}

	/**
	 * Copy a picture into a texture of its own and stand it in the scene at `tick` with side `side` (tick -1: the
	 * finished image). With an encoder, the copy is recorded there (in order with the work that made the picture).
	 */
	private keep(src: GPUTexture, tick: number, side: number, enc?: GPUCommandEncoder) {
		const tex = this.device.createTexture({
			size: [src.width, src.height],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
		});
		this.textures.push(tex);
		const e = enc ?? this.device.createCommandEncoder();
		e.copyTextureToTexture({ texture: src }, { texture: tex }, [src.width, src.height]);
		if (!enc) this.device.queue.submit([e.finish()]);
		const plane = this.planes.add(tex);
		if (tick >= 0) this.canvases.push({ plane, tick, side });
		return plane;
	}

	/**
	 * Where the names of the words go at `cut` (in ticks), and how bright: while the painter works, in the order of the
	 * prompt in a column or two just ahead of the front (`gap` apart: a name's height), each as bright as its thread
	 * reaches out of the cable towards the reading; under their maps beside the finished picture at the end.
	 */
	names(cut: number, gap = 0.3): { pos: V3; light: number }[] {
		if (this.done && cut >= this.ticksTotal) {
			const f = this.finale.side,
				c = this.finalAt,
				light = this.finaleIn;
			return this.finale.spots.map((q) => ({
				pos: [c[0] - q.z, c[1] + q.y - f / 2 - 0.45, 0.05] as V3,
				light
			}));
		}
		const n = this.wordCount,
			at = Math.max(0, cut);
		const tips = Array.from({ length: n }, (_, k) => this.weave.tip(k, at));
		const x = Math.max(this.xOf(Math.round(at)) + BLOCK / 2, ...tips.map((p) => p[0])) + 0.4;
		const cols = n > 14 ? 2 : 1,
			rows = Math.ceil(n / cols);
		const top = (READINGS + CABLE[1]) / 2 + ((rows - 1) * gap) / 2 + BLOCK / 2;
		return tips.map((_, k) => {
			const c = Math.floor(k / rows),
				r = k % rows;
			return {
				pos: [x + c * gap * 5, top - r * gap, CABLE[2]] as V3,
				light: this.weave.reachAt(k, at)
			};
		});
	}

	private get finaleIn() {
		const t = Math.min(1, (performance.now() - this.doneAt) / 1200);
		return t * t * (3 - 2 * t);
	}

	/** Place the pictures for `cut` (in ticks; nothing after it is shown); they are drawn with the scene's others. */
	place(cut: number) {
		const end = this.done && cut >= this.ticksTotal;
		const c = this.finalAt;
		this.mapPlanes.forEach((plane, k) => {
			const q = this.finale.spots[k],
				f = this.finale.side;
			this.planes.place(
				plane,
				...card(c[0] - q.z, c[1] + q.y - f / 2, f),
				end ? this.finaleIn : 0,
				1.5,
				1
			);
		});
		for (const k of this.canvases)
			this.planes.place(
				k.plane,
				...card(this.xOf(k.tick), PICTURES, k.side),
				cut >= k.tick ? 1 : 0,
				fade(cut - k.tick)
			);
		if (this.finalPlane >= 0)
			this.planes.place(this.finalPlane, ...card(c[0], c[1] - FINAL / 2, FINAL), end ? 1 : 0);
	}

	/** Draw every picture and reading up to `cut` (with the other pictures, before the lines). */
	drawGallery(pass: GPURenderPassEncoder, cut: number) {
		this.gallery.draw(pass, cut);
	}

	/** Draw the braid (as far as the text side has run, and no further than `cut`, 0..1 of the handoff) and threads. */
	drawLines(pass: GPURenderPassEncoder, w: number, h: number, cut = 1, gain = 1, paintCut = -1) {
		if (gain <= 0.01) return;
		this.weave.draw(pass, w, h, paintCut, gain);
		this.braid.now = Math.min(this.encoded, cut);
		this.braid.width = 0.8;
		this.braid.gain = 0.8 * gain;
		this.braid.fresh = 2;
		this.braid.draw(pass, w, h);
	}

	destroy() {
		this.weave.destroy();
		this.braid.destroy();
		this.gallery.destroy();
		for (const t of this.textures) t.destroy();
	}
}

/** Attention over 16 x 16 patches [patch][word] as over 32 x 32 (each patch repeated over the 2 x 2 it covers). */
function enlarge(a: Float32Array, n: number) {
	const out = new Float32Array(GRID * GRID * n),
		g = GRID / 2;
	for (let y = 0; y < GRID; y++)
		for (let x = 0; x < GRID; x++)
			out.set(
				a.subarray(((y >> 1) * g + (x >> 1)) * n, ((y >> 1) * g + (x >> 1) + 1) * n),
				(y * GRID + x) * n
			);
	return out;
}

/**
 * Where a map (32 x 32, row 0 at the top) peaks, as a fraction across and down: the brightest cell of the map blurred
 * over 3 x 3 cells, refined to the centre of the light around it.
 */
function peak(m: Float32Array): Omit<Stitch, 'share'> {
	const at = (x: number, y: number) =>
		m[Math.min(GRID - 1, Math.max(0, y)) * GRID + Math.min(GRID - 1, Math.max(0, x))];
	const blur = (x: number, y: number) => {
		let v = 0;
		for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v += at(x + dx, y + dy);
		return v;
	};
	let bx = 0,
		by = 0,
		best = -1;
	for (let y = 0; y < GRID; y++)
		for (let x = 0; x < GRID; x++) {
			const v = blur(x, y);
			if (v > best) [best, bx, by] = [v, x, y];
		}
	let sx = 0,
		sy = 0,
		sw = 0;
	for (let dy = -1; dy <= 1; dy++)
		for (let dx = -1; dx <= 1; dx++) {
			const x = Math.min(GRID - 1, Math.max(0, bx + dx)),
				y = Math.min(GRID - 1, Math.max(0, by + dy)),
				v = blur(x, y);
			sx += v * x;
			sy += v * y;
			sw += v;
		}
	return sw > 0
		? { u: (sx / sw + 0.5) / GRID, v: (sy / sw + 0.5) / GRID }
		: { u: (bx + 0.5) / GRID, v: (by + 0.5) / GRID };
}

/**
 * Places for n maps beside a finished picture of the given side: rows either side of it, filled in reading order
 * (a row runs across the left group, then the right one), as large as fits (up to a fifth of the picture's side).
 * y is up from the picture's centre, z is to the left of it.
 */
function beside(side: number, n: number) {
	const gap = side * 0.03;
	let m = side * 0.19,
		rows = 1,
		cols = 1;
	for (; m > side * 0.04; m *= 0.94) {
		rows = Math.max(1, Math.floor((side + gap) / (m + gap)));
		cols = Math.ceil(n / (2 * rows));
		if (cols * (m + gap) <= side * 0.75) break;
	}
	const pitch = rows > 1 ? (side - m) / (rows - 1) : 0;
	const spots = Array.from({ length: n }, (_, k) => {
		const row = Math.floor(k / (2 * cols)),
			j = k % (2 * cols);
		const y = side / 2 - m / 2 - row * pitch;
		const z =
			j < cols
				? side / 2 + gap * 2 + m / 2 + (cols - 1 - j) * (m + gap)
				: -(side / 2 + gap * 2 + m / 2 + (j - cols) * (m + gap));
		return { y, z };
	});
	return { spots, side: m };
}
