// The painter in the compute cube: every multiplication of a painting, box by box, below the reader's. As for the
// reader, a matrix times the rows is a box (across: the matrix's inputs; down: its outputs; in depth: the rows), but
// the painter's rows are all 1,536 of the joint stream, the prompt's 512 (its tokens, then padding) and the
// picture's 1,024 patches, and the whole of it runs once per step.
//   first, once: the reader reads the prompt again, padded to 512 tokens, through 21 layers; the adapter takes each
//   token's three states (6,144 numbers) to 7,680; the context embedder (dense) takes those to the painter's 3,072
//   then, each step: the patch embedder (dense, 128 latent channels to 3,072); 5 double blocks, whose boxes hold the
//   words' weights in their first 512 rows of depth and the picture's in the rest; 20 single blocks (one matrix for
//   q, k, v, gate and up; one back); attention in each (every row against every row: 1,536 x 1,536 x 3,072, both
//   ways); and the output projection (dense, 3,072 to 128) that gives each patch its velocity.
// The painting keeps the inputs of every matrix only for the prompt's words and 17 followed patches (see
// PainterCapture), so those are the slices lit inside each box, at their own depth; the box around them is every
// row, all computed. At the step it keeps in full, each kept row's attention shares light the attention box.
import { Painter } from '$lib/runtime/painter';
import type { Panels } from './panels';
import type { PaintingRun } from './painting-run';
import type { V3 } from './stage';

const { D, H, HD, MLP, NT, NI, NJ, CIN, CTX } = Painter.dims;
const U = 256;
const GAP = 3;
const BLOCK_GAP = 8;
const READER = { dim: 2048, layers: 21, perLayer: 50_331_648 }; // weights per reader layer (q, k, v, o, gate, up, down)

export interface PainterBox {
	name: string;
	layer: number;
	lo: V3;
	hi: V3;
	flops: number;
	slices: number[];
	faceIn: number[];
	faceOut: number[];
	K: number;
	painter: { step: number; block: number };
	hull: number[];
}

interface SliceGain {
	s: number;
	b: number; // -1 before the blocks, 25 after them
	row: number; // a kept row whose input sets the brightness
	input: string;
	n: number;
	weight: number; // typical weight size
	attn?: boolean;
}

export class PainterCube {
	/** Pairs (row, input shift) for the slices' tables, written to the index buffer from tableBase. */
	private tables: number[] = [];
	private gainOf = new Map<number, SliceGain>();
	private lit = new Set<number>();
	private meanScale = new Map<string, number>();
	private denseMean = new Map<string, number>();
	width = 0;
	height = 0;
	flops = 0;
	readerFlops = 0;
	attnFlops = 0;
	/** Where each step's row starts (for labels). */
	stepY: number[] = [];
	blockX: number[] = [];

	constructor(
		private painter: Painter,
		public run: PaintingRun,
		private tableBase: number
	) {}

	/** The typical size of each ternary weight's scales, and of each dense weight (sampled). */
	async prepare() {
		const W = this.painter.weights;
		const names = W.names.filter((n) => n.includes('blocks') || n === 'adapter.weight');
		const SAMPLE = 1024;
		const all = await Promise.all(
			names.map((n) => this.painter.readFloats('scales', W.ternary(n).scales, SAMPLE))
		);
		names.forEach((n, i) => {
			let s = 0;
			for (const a of all[i]) s += Math.abs(a) / SAMPLE;
			this.meanScale.set(n, s * 0.61); // times the share of weights that are not 0
		});
		for (const n of ['context_embedder.weight', 'x_embedder.weight', 'proj_out.weight']) {
			const v = await this.painter.readFloats('dense', W.dense_at(n), 16384);
			let s = 0;
			for (const a of v) s += Math.abs(a) / v.length;
			this.denseMean.set(n, s);
		}
	}

	/** Add every box of the painting (below the reader, from y0); box() adds each to the cube's timeline. */
	layout(
		P: Panels,
		o: {
			x0: number;
			y0: number;
			dz: number;
			box: (b: PainterBox, dur: number) => void;
		}
	) {
		this.tables = [];
		this.gainOf.clear();
		this.lit.clear();
		this.stepY = [];
		this.blockX = [];
		this.flops = 0;
		this.attnFlops = 0;
		const run = this.run,
			cap = run.cap,
			W = this.painter.weights,
			dz = o.dz;
		const words = run.words,
			patches = run.opts.patches.map((p) => NT + p),
			all = [...words, ...patches];
		const depth = NJ * dz;
		// a table of the kept rows that have this input: pairs (row, shift from the smallest offset)
		const table = (rows: number[], at: (r: number) => number | undefined) => {
			const got = rows
				.map((r) => [r, at(r)] as const)
				.filter((x): x is readonly [number, number] => x[1] !== undefined);
			if (!got.length) return null;
			const base = Math.min(...got.map((x) => x[1]));
			const t = this.tableBase + this.tables.length;
			for (const [r, x] of got) this.tables.push(r, x - base);
			return { t, count: got.length, base, first: got[0][0] };
		};
		type Tab = NonNullable<ReturnType<typeof table>>;
		// the slices add up along the view (up to 34 at once): each gives off a share of the light
		const sliceAlpha = 0.07;
		// one slice per kept row: a ternary matrix times that row's input
		const slices = (
			name: string,
			origin: V3,
			tab: Tab | null,
			input: string,
			s: number,
			b: number
		): number[] => {
			if (!tab) return [];
			const t = W.ternary(name);
			const panel = P.add({
				origin,
				ax: [t.cols / U, 0, 0],
				ay: [0, -t.rows / U, 0],
				rows: t.rows,
				cols: t.cols,
				kind: 'ternary',
				tensor: { ...t, meanScale: 0 },
				painter: true,
				x: { offset: tab.base, cap: true },
				instances: { count: tab.count, step: [0, 0, dz], table: tab.t },
				gain: 1,
				additive: true,
				alpha: sliceAlpha,
				samples: 2,
				grid: false,
				lift: 2,
				visible: false
			});
			this.gainOf.set(panel, {
				s,
				b,
				row: tab.first,
				input,
				n: t.cols,
				weight: this.meanScale.get(name) ?? 0.01
			});
			return [panel];
		};
		// a dense matrix (rows x cols, row-major) times each kept row's input
		const denseSlices = (
			name: string,
			origin: V3,
			rows: number,
			cols: number,
			tab: Tab | null,
			input: string,
			s: number,
			b: number
		): number[] => {
			if (!tab) return [];
			const panel = P.add({
				origin,
				ax: [cols / U, 0, 0],
				ay: [0, -rows / U, 0],
				rows,
				cols,
				kind: 'dense',
				a: { offset: W.dense_at(name), rs: cols, cs: 1, dw: true },
				b: { offset: tab.base, rs: 0, cs: 1, cap: true },
				instances: { count: tab.count, step: [0, 0, dz], table: tab.t },
				gain: 1,
				additive: true,
				alpha: sliceAlpha,
				samples: 2,
				grid: false,
				lift: 2,
				visible: false
			});
			this.gainOf.set(panel, {
				s,
				b,
				row: tab.first,
				input,
				n: cols,
				weight: this.denseMean.get(name) ?? 0.01
			});
			return [panel];
		};
		// the edges of a box (its top, left and front outlined), so its size shows however little is lit
		const hull = (lo: V3, hi: V3) => {
			const frame = (origin: V3, ax: V3, ay: V3) =>
				P.add({
					origin,
					ax,
					ay,
					rows: 1,
					cols: 1,
					kind: 'frame',
					gain: 1,
					additive: true,
					alpha: 0.16
				});
			const w = hi[0] - lo[0],
				h = hi[1] - lo[1],
				d = hi[2] - lo[2];
			return [
				frame([lo[0], hi[1], lo[2]], [w, 0, 0], [0, 0, d]),
				frame([lo[0], hi[1], lo[2]], [0, 0, d], [0, -h, 0]),
				frame([lo[0], hi[1], hi[2]], [w, 0, 0], [0, -h, 0])
			];
		};
		const box = (
			name: string,
			s: number,
			b: number,
			lo: V3,
			hi: V3,
			flops: number,
			sl: number[],
			K: number,
			dur: number
		) => {
			o.box(
				{
					name,
					layer: -2,
					lo,
					hi,
					flops,
					slices: sl,
					faceIn: [],
					faceOut: [],
					K,
					painter: { step: s, block: b },
					hull: hull(lo, hi)
				},
				dur
			);
			this.flops += flops;
		};

		let x = o.x0;
		let y = o.y0;
		// ---- once: the reader again (512 tokens, 21 layers), the adapter, the context embedder
		{
			const w = READER.layers * 3,
				h = 40;
			const rd = NT * dz;
			this.readerFlops =
				2 * READER.perLayer * NT * READER.layers +
				4 * NT * words.length * READER.dim * READER.layers;
			box(
				'the reader again: the prompt padded to 512 tokens, 21 layers (not kept here)',
				-1,
				-2,
				[x, y - h, 0],
				[x + w, y, rd],
				this.readerFlops,
				[],
				1,
				1
			);
			x += w + GAP * 2;
			const a = W.ternary('adapter.weight');
			const ta = table(words, (r) => cap.at(-1, -1, r, 'taps'));
			const sa = slices('adapter.weight', [x, y, 0], ta, 'taps', -1, -1);
			box(
				'the adapter (ternary)',
				-1,
				-1,
				[x, y - a.rows / U, 0],
				[x + a.cols / U, y, rd],
				2 * a.rows * a.cols * (NT - 3),
				sa,
				a.cols,
				1
			);
			x += a.cols / U + GAP;
			const tc = table(words, (r) => cap.at(-1, -1, r, 'ctx'));
			const sc = denseSlices('context_embedder.weight', [x, y, 0], D, CTX, tc, 'ctx', -1, -1);
			box(
				'the context embedder (dense)',
				-1,
				-1,
				[x, y - D / U, 0],
				[x + CTX / U, y, rd],
				2 * D * CTX * NT,
				sc,
				CTX,
				1
			);
			x += CTX / U;
			y -= Math.max(a.rows / U, h) + 30;
		}
		// ---- each step
		const rowH = Math.max((3 * D + 2 * MLP) / U, depth) + 40;
		let widest = x - o.x0;
		for (let s = 0; s < run.steps; s++) {
			const y0 = y - s * rowH;
			this.stepY.push(y0);
			x = o.x0;
			// the patches into the painter
			{
				const tl = table(patches, (r) => cap.at(s, -1, r, 'lat'));
				const sl = denseSlices('x_embedder.weight', [x, y0, 0], D, CIN, tl, 'lat', s, -1);
				box(
					'the patch embedder (dense)',
					s,
					-1,
					[x, y0 - D / U, 0],
					[x + CIN / U, y0, depth],
					2 * D * CIN * NI,
					sl,
					CIN,
					0.3
				);
				x += CIN / U + GAP;
			}
			for (let b = 0; b < 25; b++) {
				if (s === 0) this.blockX.push(x);
				const dbl = b < 5;
				const at = (name: string) => (r: number) => cap.at(s, b, r, name);
				const attention = () => {
					const full = s === run.fullStep;
					const tab = full ? table(all, at('attn')) : null;
					const sl: number[] = [];
					if (tab) {
						const panel = P.add({
							origin: [x, y0, 0],
							ax: [D / U, 0, 0],
							ay: [0, -depth, 0],
							rows: NJ,
							cols: D,
							kind: 'dense',
							// the share of source row r (down) for the head of column c, for this kept row (the slice)
							a: { offset: tab.base, rs: 1, cs: 0, div: HD, ds: NJ, cap: true },
							instances: { count: tab.count, step: [0, 0, dz], table: tab.t, tableA: true },
							gain: 1,
							additive: true,
							alpha: sliceAlpha,
							samples: 2,
							grid: false,
							lift: 0.3,
							visible: false
						});
						this.gainOf.set(panel, {
							s,
							b,
							row: tab.first,
							input: 'attn',
							n: H * NJ,
							weight: 1,
							attn: true
						});
						sl.push(panel);
					}
					const fl = 2 * (2 * NJ * NJ * D);
					this.attnFlops += fl;
					box(
						full
							? 'attention: every row against every row (the kept rows show their shares)'
							: 'attention: every row against every row',
						s,
						b,
						[x, y0 - depth, 0],
						[x + D / U, y0, depth],
						fl,
						sl,
						D,
						0.3
					);
					x += D / U + GAP;
				};
				if (dbl) {
					const pre = `transformer_blocks.${b}.`;
					const nm = {
						txt: {
							q: 'attn.add_q_proj',
							k: 'attn.add_k_proj',
							v: 'attn.add_v_proj',
							o: 'attn.to_add_out',
							in: 'ff_context.linear_in',
							out: 'ff_context.linear_out'
						},
						img: {
							q: 'attn.to_q',
							k: 'attn.to_k',
							v: 'attn.to_v',
							o: 'attn.to_out.0',
							in: 'ff.linear_in',
							out: 'ff.linear_out'
						}
					};
					const both = (key: keyof typeof nm.txt, input: string, origin: V3) =>
						(['txt', 'img'] as const).flatMap((st) =>
							slices(
								`${pre}${nm[st][key]}.weight`,
								origin,
								table(st === 'txt' ? words : patches, at(input)),
								input,
								s,
								b
							)
						);
					// q, k, v
					{
						const sl = [0, 1, 2].flatMap((i) =>
							both((['q', 'k', 'v'] as const)[i], 'n1', [x, y0 - i * (D / U + 0.2), 0])
						);
						const h = (3 * D) / U + 0.4;
						box(
							"q, k and v (the words' weights in the first 512 rows, the picture's in the rest)",
							s,
							b,
							[x, y0 - h, 0],
							[x + D / U, y0, depth],
							2 * 3 * D * D * NJ,
							sl,
							D,
							0.3
						);
						x += D / U + GAP;
					}
					attention();
					{
						const sl = both('o', 'o', [x, y0, 0]);
						box(
							'output projection',
							s,
							b,
							[x, y0 - D / U, 0],
							[x + D / U, y0, depth],
							2 * D * D * NJ,
							sl,
							D,
							0.3
						);
						x += D / U + GAP;
					}
					{
						const sl = both('in', 'n2', [x, y0, 0]);
						box(
							'gate and up',
							s,
							b,
							[x, y0 - (2 * MLP) / U, 0],
							[x + D / U, y0, depth],
							2 * 2 * MLP * D * NJ,
							sl,
							D,
							0.3
						);
						x += D / U + GAP;
					}
					{
						const sl = both('out', 'cat', [x, y0, 0]);
						box(
							'down',
							s,
							b,
							[x, y0 - D / U, 0],
							[x + MLP / U, y0, depth],
							2 * D * MLP * NJ,
							sl,
							MLP,
							0.3
						);
						x += MLP / U + BLOCK_GAP;
					}
				} else {
					const pre = `single_transformer_blocks.${b - 5}.attn.`;
					{
						const sl = slices(
							pre + 'to_qkv_mlp_proj.weight',
							[x, y0, 0],
							table(all, at('n1')),
							'n1',
							s,
							b
						);
						const M = 3 * D + 2 * MLP;
						box(
							'q, k, v, gate and up: one matrix',
							s,
							b,
							[x, y0 - M / U, 0],
							[x + D / U, y0, depth],
							2 * M * D * NJ,
							sl,
							D,
							0.3
						);
						x += D / U + GAP;
					}
					attention();
					{
						const sl = slices(
							pre + 'to_out.weight',
							[x, y0, 0],
							table(all, at('cat')),
							'cat',
							s,
							b
						);
						box(
							"one matrix back (attention's result and the neurons together)",
							s,
							b,
							[x, y0 - D / U, 0],
							[x + (D + MLP) / U, y0, depth],
							2 * D * (D + MLP) * NJ,
							sl,
							D + MLP,
							0.3
						);
						x += (D + MLP) / U + BLOCK_GAP;
					}
				}
			}
			// each patch's velocity
			{
				const tn = table(patches, (r) => cap.at(s, 25, r, 'nout'));
				const sl = denseSlices('proj_out.weight', [x, y0, 0], CIN, D, tn, 'nout', s, 25);
				box(
					'the output projection: each patch’s velocity (dense)',
					s,
					25,
					[x, y0 - CIN / U, 0],
					[x + D / U, y0, depth],
					2 * CIN * D * NI,
					sl,
					D,
					0.3
				);
				x += D / U;
			}
			widest = Math.max(widest, x - o.x0);
		}
		this.width = widest;
		this.height = o.y0 - (y - (run.steps - 1) * rowH - rowH);
		return new Uint32Array(this.tables);
	}

	/** Whether a box's inputs have been computed yet. */
	ready(p: { step: number; block: number }) {
		const run = this.run;
		if (p.step < 0) return run.has(0, 0);
		if (p.block < 0) return run.has(p.step, 0);
		return run.has(p.step, Math.min(p.block, 25));
	}

	/** Brightness of the slices whose inputs have arrived since last time (from a kept row's input). */
	async refresh(P: Panels) {
		const todo: [number, SliceGain, { offset: number; count: number }][] = [];
		for (const [panel, g] of this.gainOf) {
			if (this.lit.has(panel)) continue;
			if (!this.ready({ step: g.s, block: g.b })) continue;
			const s = g.s < 0 ? -1 : g.s;
			const off = this.run.cap.at(s, g.b, g.row, g.input);
			if (off === undefined) continue;
			todo.push([panel, g, { offset: off, count: g.n }]);
		}
		if (!todo.length) return;
		for (const [panel] of todo) this.lit.add(panel);
		const got = await this.run.read(todo.map((t) => t[2]));
		todo.forEach(([panel, g], i) => {
			let m = 0;
			for (const a of got[i]) m += Math.abs(a) / got[i].length;
			// shares: a typical share is 1 / 1,536; the rest: typical weight times typical input
			P.set(panel, { gain: g.attn ? 10 / NJ : g.weight * m * 2.2 || 1 });
		});
	}
}
