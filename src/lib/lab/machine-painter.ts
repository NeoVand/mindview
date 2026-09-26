// The painter's part of the machine: every weight of Bonsai Image 4B's DiT (3.7 billion, ternary) and the adapter,
// on the same wall as the reader, one cell each, in the order a painting uses them.
//   the adapter: takes each word's three states from the reader (layers 7, 14, 21: 6,144 numbers) to 7,680; then
//     the context embedder (dense) to the painter's 3,072
//   5 double blocks: two columns each, the words' weights (text) and the picture's (image): q, k, v, the output
//     projection, and the MLP (gate and up fused, then down); the two streams meet only inside attention
//   20 single blocks: one column each, words and patches through the same weights: q, k, v and the MLP's gate and up
//     in one matrix, then one output matrix
//   the output projection (dense): each patch's 3,072 numbers to its 128 latent channels, the velocity
// Above each block, the picture it has in mind (for the chosen step); at the end, the finished picture.
// Lit by a painting's capture: for the chosen word (or patch) and step, each matrix's input, so each cell shows its
// weight times the number it multiplies. A word goes through the text weights, a patch through the image weights;
// in single blocks both through the same.
import type { Painter } from '$lib/runtime/painter';
import type { ImagePlanes } from '$lib/viz/planes';
import type { Panels } from './panels';
import type { PaintingRun } from './painting-run';
import type { V3 } from './stage';

const U = 256;
const GAP = 0.5;
const COL_GAP = 2.5;
const WAVE = 1.6;
const WAVE_GAP = 0.35;

export interface PMat {
	panel: number;
	name: string;
	label: string;
	block: number; // -1 adapter and embedders, 0..24, 25 output projection
	stream: 'txt' | 'img' | 'joint';
	input: string; // capture name of its input (see PainterCapture)
	dense: boolean;
	origin: V3;
	w: number;
	h: number;
}

export interface PWave {
	panel: number;
	block: number;
	stream: 'txt' | 'img' | 'joint';
	input: string;
	n: number;
}

export class PainterWall {
	readonly mats: PMat[] = [];
	readonly waves: PWave[] = [];
	private pics: { block: number; origin: V3; size: number }[] = [];
	private planeOf = new Map<GPUTexture, number>();
	private finalPlane = -1;
	private finalAt: V3 = [0, 0, 0];
	readonly x0: number;
	x1 = 0;
	height = 0;
	private blockX: number[] = [];
	finalSize = 150;

	constructor(
		private panels: Panels,
		private planes: ImagePlanes,
		private painter: Painter,
		x0: number
	) {
		this.x0 = x0;
		const { D, MLP, CTX, TAP, CIN } = (painter.constructor as typeof Painter).dims;
		const W = painter.weights;
		const P = panels;
		let x = x0;
		const col = D / U; // 12 units
		const mat = (
			name: string,
			label: string,
			block: number,
			stream: PMat['stream'],
			input: string,
			origin: V3,
			transpose: boolean
		) => {
			const t = W.ternary(name);
			const rows = transpose ? t.cols : t.rows,
				cols = transpose ? t.rows : t.cols;
			const w = cols / U,
				h = rows / U;
			const panel = P.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -h, 0],
				rows,
				cols,
				kind: 'ternary',
				tensor: { rows: t.rows, cols: t.cols, codes: t.codes, scales: t.scales, meanScale: 0 },
				painter: true,
				transpose,
				gain: 1,
				pyramid: true,
				digits: true,
				lift: 1.5
			});
			this.mats.push({ panel, name, label, block, stream, input, dense: false, origin, w, h });
			return { w, h };
		};
		const dense = (
			name: string,
			label: string,
			block: number,
			stream: PMat['stream'],
			input: string,
			origin: V3,
			rows: number,
			cols: number
		) => {
			const w = cols / U,
				h = rows / U;
			const panel = P.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -h, 0],
				rows,
				cols,
				kind: 'dense',
				a: { offset: W.dense_at(name), rs: cols, cs: 1, dw: true },
				gain: 1,
				digits: true,
				lift: 1.5,
				samples: 3
			});
			this.mats.push({ panel, name, label, block, stream, input, dense: true, origin, w, h });
			return { w, h };
		};
		const wave = (
			block: number,
			stream: PWave['stream'],
			input: string,
			origin: V3,
			ax: V3,
			ay: V3,
			n: number
		) => {
			const panel = P.add({
				origin,
				ax,
				ay,
				rows: 1,
				cols: n,
				kind: 'bars',
				a: { offset: 0, rs: 0, cs: 1, cap: true },
				gain: 1,
				digits: true,
				alpha: 0
			});
			this.waves.push({ panel, block, stream, input, n });
		};

		// ---- the adapter and the embedders
		{
			const x1 = x + WAVE + WAVE_GAP;
			// the reader's three states along the left, the adapter's 7,680 outputs along the top of the embedder
			wave(-1, 'txt', 'taps', [x1 - WAVE_GAP, 0, 0], [0, -TAP / U, 0], [-WAVE, 0, 0], TAP);
			const a = mat(
				'adapter.weight',
				'the adapter (transposed)',
				-1,
				'txt',
				'taps',
				[x1, 0, 0],
				true
			);
			let y = -a.h - GAP;
			wave(-1, 'txt', 'ctx', [x1, y, 0], [CTX / U, 0, 0], [0, -WAVE, 0], CTX);
			y -= WAVE + WAVE_GAP;
			dense('context_embedder.weight', 'context embedder', -1, 'txt', 'ctx', [x1, y, 0], D, CTX);
			y -= D / U + GAP * 4;
			// the patch's latent (128 channels) into the painter's 3,072
			dense('x_embedder.weight', 'patch embedder', -1, 'img', 'lat', [x1, y, 0], D, CIN);
			wave(-1, 'img', 'lat', [x1, y + WAVE + WAVE_GAP, 0], [CIN / U, 0, 0], [0, -WAVE, 0], CIN);
			x = x1 + CTX / U + COL_GAP * 2;
		}
		// ---- double blocks: text column, image column
		for (let b = 0; b < 5; b++) {
			const pre = `transformer_blocks.${b}.`;
			this.blockX.push(x + col + WAVE + COL_GAP / 2);
			for (const st of ['txt', 'img'] as const) {
				const cx = x + WAVE + WAVE_GAP;
				const n = (s: string, t: string) => pre + (st === 'txt' ? s : t);
				let y = 0;
				wave(b, st, 'n1', [cx, y + WAVE + WAVE_GAP, 0], [col, 0, 0], [0, -WAVE, 0], D);
				for (const [s, t, l] of [
					['attn.add_q_proj.weight', 'attn.to_q.weight', 'q'],
					['attn.add_k_proj.weight', 'attn.to_k.weight', 'k'],
					['attn.add_v_proj.weight', 'attn.to_v.weight', 'v']
				]) {
					const m = mat(
						n(s, t),
						`${l} (${st === 'txt' ? 'words' : 'picture'})`,
						b,
						st,
						'n1',
						[cx, y, 0],
						false
					);
					y -= m.h + GAP;
				}
				wave(b, st, 'o', [cx - WAVE_GAP, y, 0], [0, -D / U, 0], [-WAVE, 0, 0], D);
				const o = mat(
					n('attn.to_add_out.weight', 'attn.to_out.0.weight'),
					`output (${st === 'txt' ? 'words' : 'picture'})`,
					b,
					st,
					'o',
					[cx, y, 0],
					true
				);
				y -= o.h + GAP;
				wave(b, st, 'n2', [cx, y, 0], [col, 0, 0], [0, -WAVE * 0.8, 0], D);
				y -= WAVE * 0.8 + WAVE_GAP * 0.4;
				const li = mat(
					n('ff_context.linear_in.weight', 'ff.linear_in.weight'),
					`gate | up (${st === 'txt' ? 'words' : 'picture'})`,
					b,
					st,
					'n2',
					[cx, y, 0],
					false
				);
				y -= li.h + GAP;
				wave(b, st, 'cat', [cx - WAVE_GAP, y, 0], [0, -MLP / U, 0], [-WAVE, 0, 0], MLP);
				const lo = mat(
					n('ff_context.linear_out.weight', 'ff.linear_out.weight'),
					`down (${st === 'txt' ? 'words' : 'picture'})`,
					b,
					st,
					'cat',
					[cx, y, 0],
					true
				);
				y -= lo.h;
				this.height = Math.max(this.height, -y);
				x = cx + col + COL_GAP;
			}
			x += COL_GAP;
		}
		// ---- single blocks
		for (let b = 0; b < 20; b++) {
			const pre = `single_transformer_blocks.${b}.`;
			const bb = 5 + b;
			const cx = x + WAVE + WAVE_GAP;
			this.blockX.push(cx + col / 2);
			let y = 0;
			wave(bb, 'joint', 'n1', [cx, y + WAVE + WAVE_GAP, 0], [col, 0, 0], [0, -WAVE, 0], D);
			const qm = mat(
				pre + 'attn.to_qkv_mlp_proj.weight',
				'q | k | v | gate | up',
				bb,
				'joint',
				'n1',
				[cx, y, 0],
				false
			);
			y -= qm.h + GAP;
			wave(
				bb,
				'joint',
				'cat',
				[cx - WAVE_GAP, y, 0],
				[0, -(D + MLP) / U, 0],
				[-WAVE, 0, 0],
				D + MLP
			);
			const o = mat(
				pre + 'attn.to_out.weight',
				'output (transposed)',
				bb,
				'joint',
				'cat',
				[cx, y, 0],
				true
			);
			y -= o.h;
			this.height = Math.max(this.height, -y);
			x = cx + col + COL_GAP;
		}
		// ---- the output projection (each patch's velocity), and the finished picture
		{
			const cx = x + COL_GAP;
			wave(25, 'img', 'nout', [cx, WAVE + WAVE_GAP, 0], [col, 0, 0], [0, -WAVE, 0], D);
			dense('proj_out.weight', 'output projection', 25, 'img', 'nout', [cx, 0, 0], CIN, D);
			x = cx + col + COL_GAP * 3;
			this.finalAt = [x, 0, 0];
			this.finalPlane = planes.add(painter.taef2.texture);
			x += this.finalSize;
		}
		this.x1 = x;
		// the pictures above the blocks (textures arrive as it paints)
		this.blockX.forEach((bx, b) => {
			const size = b < 5 ? 2 * col : col;
			this.pics.push({ block: b, origin: [bx - size / 2, 4 + size, 0], size });
		});
	}

	/** The x of a block's middle. */
	blockCentre(b: number) {
		return this.blockX[b];
	}

	/** Point every panel at the chosen row (a word's or a patch's joint row) and step; returns the panels lit. */
	light(run: PaintingRun | null, step: number, row: number, work: boolean): number[] {
		const P = this.panels;
		const lit: number[] = [];
		const cap = run?.cap;
		const isWord = row < 512;
		const uses = (st: PMat['stream']) => st === 'joint' || (st === 'txt') === isWord;
		const offsetOf = (block: number, input: string) => {
			if (!cap) return undefined;
			if (block === -1 && (input === 'taps' || input === 'ctx')) return cap.at(-1, -1, row, input);
			if (block === -1 && input === 'lat') return cap.at(step, -1, row, 'lat');
			return cap.at(step, block, row, input);
		};
		// an input is there once the painting has passed where it is kept: the words' states once the first block
		// has run, a patch's latent once its step has begun, a block's inputs once the block has run
		const reached = (block: number, input: string) =>
			run !== null &&
			(block >= 0
				? run.has(step, Math.min(24, block))
				: input === 'lat'
					? run.has(step, 0)
					: run.wordsKept);
		for (const m of this.mats) {
			const off = uses(m.stream) && work ? offsetOf(m.block, m.input) : undefined;
			const ready = off !== undefined && reached(m.block, m.input);
			if (m.dense) {
				P.set(m.panel, ready ? { b: { offset: off!, rs: 0, cs: 1, cap: true } } : { b: undefined });
			} else P.set(m.panel, { x: ready ? { offset: off!, cap: true } : null });
			if (ready) lit.push(m.panel);
		}
		for (const w of this.waves) {
			const off = uses(w.stream) ? offsetOf(w.block, w.input) : undefined;
			const ready = off !== undefined && reached(w.block, w.input);
			P.set(w.panel, { a: { offset: off ?? 0, rs: 0, cs: 1, cap: true }, alpha: ready ? 1 : 0 });
		}
		return lit;
	}

	/** Forget the pictures of an old painting. */
	forgetPictures() {
		this.planeOf.clear();
	}

	/** The capture regions the lit waveforms read (for their brightness). */
	waveRegions(run: PaintingRun, step: number, row: number) {
		const out: { panel: number; offset: number; count: number }[] = [];
		const isWord = row < 512;
		for (const w of this.waves) {
			if (!(w.stream === 'joint' || (w.stream === 'txt') === isWord)) continue;
			const off =
				w.block === -1 && (w.input === 'taps' || w.input === 'ctx')
					? run.cap.at(-1, -1, row, w.input)
					: w.block === -1
						? run.cap.at(step, -1, row, 'lat')
						: run.cap.at(step, w.block, row, w.input);
			if (off !== undefined) out.push({ panel: w.panel, offset: off, count: w.n });
		}
		return out;
	}

	/** Place the pictures: each block's for the chosen step (as far as it has painted), and the finished one. */
	place(run: PaintingRun | null, step: number) {
		for (const plane of this.planeOf.values())
			this.planes.place(plane, [0, 0, 0], [1, 0, 0], [0, -1, 0], 0);
		for (const p of this.pics) {
			const tex = run?.pictures[step]?.[p.block] ?? null;
			if (!tex) continue;
			let plane = this.planeOf.get(tex);
			if (plane === undefined) {
				plane = this.planes.add(tex);
				this.planeOf.set(tex, plane);
			}
			this.planes.place(plane, p.origin, [p.size, 0, 0], [0, -p.size, 0], 1, 1, 2);
		}
		const f = this.finalSize;
		this.planes.place(
			this.finalPlane,
			[this.finalAt[0], this.finalAt[1], 0],
			[f, 0, 0],
			[0, -f, 0],
			run?.done ? 1 : 0
		);
	}

	/** The picture (and the patch under the pointer) at a world point on the finished picture, or null. */
	patchAt(p: V3): number | null {
		const f = this.finalSize;
		const u = (p[0] - this.finalAt[0]) / f,
			v = (this.finalAt[1] - p[1]) / f;
		if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
		return Math.floor(v * 32) * 32 + Math.floor(u * 32);
	}

	/** Where a patch is on the finished picture (its centre). */
	patchPoint(patch: number): V3 {
		const f = this.finalSize;
		return [
			this.finalAt[0] + ((patch % 32) + 0.5) * (f / 32),
			this.finalAt[1] - (Math.floor(patch / 32) + 0.5) * (f / 32),
			0.01
		];
	}

	get finalCentre(): V3 {
		return [this.finalAt[0] + this.finalSize / 2, this.finalAt[1] - this.finalSize / 2, 0];
	}

	labelPoints(): { text: string; at: V3; size: number }[] {
		const out: { text: string; at: V3; size: number }[] = [];
		out.push({ text: 'the adapter', at: [this.x0 + 16, 4.2, 0], size: 1.4 });
		this.blockX.forEach((x, b) =>
			out.push({
				text: b < 5 ? `block ${b + 1}: words | picture` : `block ${b + 1}`,
				at: [x, b < 5 ? 30 : 17.5, 0],
				size: 1.4
			})
		);
		out.push({
			text: 'the picture',
			at: [this.finalAt[0] + this.finalSize / 2, 4.2, 0],
			size: 1.8
		});
		return out;
	}
}
