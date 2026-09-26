// Lab: one block of the painter, every number. One word of the prompt, or one patch of the picture, goes through one
// of the 25 blocks of Bonsai Image 4B's DiT at one step of a real painting, operation by operation, laid out left to
// right as stations the camera visits in turn (as the reader's layer is). Every number drawn is the painter's own,
// kept while it painted (see PainterCapture: every word and every followed patch keeps everything at one step), or
// computed here from those numbers exactly as the painter computes them (the normed row before it is set by the time
// step, q and k before they turn, the running sums of the matrices, one head's term from one row).
//   the row arrives; it is normed (LayerNorm) and set by the time step, x (1 + scale) + shift (the only way the noise
//   level enters); matrices read it (q, k, v); q and k are normed per head and turned by position (a patch by its row
//   and column on the picture, a word by its place in the prompt); each of 24 heads shares its attention over all
//   1,536 rows, the prompt's 512 and the picture's 1,024 (the only place words and picture meet); the shares mix the
//   values; a matrix takes the result back into the row, through a gate set by the time step; then the MLP (gate, up,
//   9,216 neurons, down), gated likewise. In the 5 double blocks words and patches have their own weights; in the 20
//   single blocks they share them, q, k, v, gate and up are one matrix, and one matrix takes attention and the
//   neurons back together.
import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
import { Painter } from '$lib/runtime/painter';
import { clean } from '$lib/engine/text';
import { ImagePlanes } from '$lib/viz/planes';
import { Panels, type PanelDesc } from './panels';
import type { PaintingRun } from './painting-run';
import type { Scene, Stage, V3 } from './stage';
import { fmt } from './machine';
import { PREFIX_WGSL, type LayerLabel } from './layer';

const { D, H, HD, MLP, NT, NJ } = Painter.dims;
const U = 384; // cells per world unit (3,072 is 8 units)
const STATION_GAP = 26;
const HB = 96;
const GRID = 32; // patches per side
const EPS = 1e-6;

type Kind = 'double' | 'single';
type Src = { offset: number; aux?: boolean; cap?: boolean };

interface Station {
	name: string;
	first: number;
	last: number;
	x: number;
	w: number;
	h: number;
	top: number;
	caption: () => string;
	sweep?: { mats: number[]; outs: { panel: number; K: number }[]; K: number };
	labels: {
		text: () => string;
		at: V3;
		size: number;
		align?: 'left' | 'center' | 'right';
		italic?: boolean;
	}[];
}

/** A matrix's running sums: rows M, inputs K, where they are in the scratch buffer, and the capture its input is. */
interface Pre {
	M: number;
	K: number;
	off: number;
	nck: number;
	input: string;
	fin: number; // where its finished sums are, from A.fin
}

// scratch layout (floats)
const A = {
	ln1: 0, // the row normed (LayerNorm, before it is set)
	sc1: D, // 1 + scale
	sh1: 2 * D, // shift
	qn: 3 * D, // q per head normed, before turning; pairs split (x_2p at p, x_2p+1 at 64 + p)
	qr: 4 * D, // q turned, likewise
	kn: 5 * D,
	kr: 6 * D,
	g1: 7 * D, // the first gate
	dA: 8 * D, // what attention adds (after the gate); in single blocks what the block adds
	ln2: 9 * D,
	sc2: 10 * D,
	sh2: 11 * D,
	g2: 12 * D,
	dM: 13 * D, // what the MLP adds (double blocks)
	silu: 14 * D, // SiLU(gate) [9216]
	map: 17 * D, // the heads' mean share of each patch [1024]
	sumd: 17 * D + 1024, // the chosen dot product's running sum [128]
	term: 17 * D + 1152, // share x value [128]
	hist: 17 * D + 1280, // SiLU scatter [96][96]
	stack: 17 * D + 1280 + HB * HB, // the row entering block 1 and after each block [26][3072]
	fin: 17 * D + 1280 + HB * HB + 26 * D, // every matrix's finished sums [at most 33,792]
	prefix: 17 * D + 1280 + HB * HB + 26 * D + 33792
};
const PREFIX_FLOATS = 4 * D * (D / 16) + 2 * MLP * (D / 16) + D * (MLP / 16); // the same for both kinds of block

export class Block implements Scene {
	panels: Panels;
	private planes: ImagePlanes;
	private prefixPipe: GPUComputePipeline;
	private prefixUniforms: GPUBuffer;
	private stations: Station[] = [];
	private kind: Kind | null = null;
	run: PaintingRun | null = null;
	block = 7;
	row = NT + 16 * GRID + 16;
	station = 0;
	/** Seconds into the current station (sweeps run in the first part of it). */
	t = 0;
	playing = false;
	busy = '';
	private waiting = false;
	private loadId = 0;
	private cpu?: Record<string, Float32Array>;
	private nums: Record<string, number> = {};
	private head = 0;
	private other = -1; // the row whose key and value are spelled out
	private p: Record<string, number> = {};
	private srcs = new Map<number, () => Src>();
	private mats = new Map<number, { key: string; pre: string; m0: number }>();
	private inspectors = new Map<
		string,
		{ mat: string; pre: string; m0: number; wts: number; prod: number; run: number }
	>();
	private rowPick = new Map<string, number>();
	private pre: Record<string, Pre> = {};
	private mod = new Map<number, Float32Array>();
	private planeOf = new Map<GPUTexture, number>();
	private pics: { which: 'before' | 'after'; at: V3; size: number }[] = [];
	private mapAt: V3[] = []; // top-left of each head's map (and the mean map last)
	private mapSize: number[] = [];
	private textAt: V3 = [0, 0, 0];
	private textCell = 0.3;

	constructor(
		private stage: Stage,
		llm: BonsaiLLM,
		private painter: Painter
	) {
		this.panels = new Panels(stage, llm, { capacity: 192, aux: A.prefix + PREFIX_FLOATS, idx: 4 });
		this.panels.setPainter(painter.weights);
		this.planes = new ImagePlanes(stage.device, stage.frame, true);
		this.prefixUniforms = stage.device.createBuffer({
			size: 8 * 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.prefixPipe = stage.device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: stage.device.createShaderModule({ label: 'running sums', code: PREFIX_WGSL }),
				entryPoint: 'prefix'
			}
		});
		this.build('single');
	}

	// ---- what is shown

	get isPatch() {
		return this.row >= NT;
	}
	get patch() {
		return this.row - NT;
	}
	get step() {
		return this.run?.fullStep ?? 0;
	}
	private tokenText(r: number) {
		return clean(this.run?.tokens[r] ?? '') || '·';
	}
	private rowName(r = this.row) {
		if (r < NT) return `“${this.tokenText(r)}”`;
		const p = r - NT;
		return `the patch at row ${Math.floor(p / GRID) + 1}, column ${(p % GRID) + 1}`;
	}
	private get what() {
		return this.isPatch ? 'patch' : 'word';
	}
	get stationNames() {
		return this.stations.map((s) => s.name);
	}
	get caption() {
		if (this.waiting || !this.cpu) return '';
		return this.stations[this.station]?.caption() ?? '';
	}
	/** The joint rows that keep everything (the prompt's words and the followed patches). */
	get fullRows(): number[] {
		const r = this.run;
		return r ? [...r.words, ...r.opts.patches.map((p) => NT + p)] : [];
	}

	private tensorName(key: string) {
		const b = this.block;
		if (b >= 5) {
			const P = `single_transformer_blocks.${b - 5}.attn.`;
			return P + (key === 'fused' ? 'to_qkv_mlp_proj.weight' : 'to_out.weight');
		}
		const img = this.isPatch;
		const n: Record<string, string> = img
			? {
					q: 'attn.to_q',
					k: 'attn.to_k',
					v: 'attn.to_v',
					o: 'attn.to_out.0',
					in: 'ff.linear_in',
					out: 'ff.linear_out'
				}
			: {
					q: 'attn.add_q_proj',
					k: 'attn.add_k_proj',
					v: 'attn.add_v_proj',
					o: 'attn.to_add_out',
					in: 'ff_context.linear_in',
					out: 'ff_context.linear_out'
				};
		return `transformer_blocks.${b}.${n[key]}.weight`;
	}

	private normName(which: 'q' | 'k') {
		const b = this.block;
		if (b >= 5) return `single_transformer_blocks.${b - 5}.attn.norm_${which}.weight`;
		return `transformer_blocks.${b}.attn.${this.isPatch ? `norm_${which}` : `norm_added_${which}`}.weight`;
	}

	private capAt(name: string, row = this.row, block = this.block) {
		const o = this.run?.cap.at(this.step, block, row, name);
		if (o === undefined) throw new Error(`not kept: ${name} of row ${row}, block ${block}`);
		return o;
	}

	// ---- layout

	private build(kind: Kind) {
		const P = this.panels;
		P.clear();
		this.stations = [];
		this.p = {};
		this.srcs.clear();
		this.mats.clear();
		this.inspectors.clear();
		this.pics = [];
		this.mapAt = [];
		this.mapSize = [];
		this.kind = kind;
		const dbl = kind === 'double';
		const specs: [string, number, number, string][] = dbl
			? [
					['q', D, D, 'n1'],
					['k', D, D, 'n1'],
					['v', D, D, 'n1'],
					['o', D, D, 'o'],
					['in', 2 * MLP, D, 'n2'],
					['out', D, MLP, 'cat']
				]
			: [
					['fused', 3 * D + 2 * MLP, D, 'n1'],
					['out', D, D + MLP, 'cat']
				];
		this.pre = {};
		let off = A.prefix,
			fin = 0;
		for (const [k, M, K, input] of specs) {
			this.pre[k] = { M, K, off, nck: K / 16, input, fin };
			off += (M * K) / 16;
			fin += M;
		}
		this.finCount = fin;
		const cap =
			(name: string, add = 0) =>
			() => ({ offset: this.capAt(name) + add, cap: true });
		const aux = (o: number) => () => ({ offset: o, aux: true });

		const wave = (
			key: string,
			origin: V3,
			len: number,
			thick: number,
			n: number,
			src: () => Src,
			extra: Partial<PanelDesc> = {}
		) => {
			const panel = P.add({
				origin,
				ax: [len, 0, 0],
				ay: [0, -thick, 0],
				rows: 1,
				cols: n,
				kind: 'bars',
				a: { offset: 0, rs: 0, cs: 1 },
				gain: 1,
				digits: true,
				lift: 1.2,
				...extra
			});
			this.p[key] = panel;
			this.srcs.set(panel, src);
			return panel;
		};
		const matrix = (key: string, pre: string, m0: number, rows: number, origin: V3) => {
			const K = this.pre[pre].K;
			const w = K / U,
				h = rows / U;
			const panel = P.add({
				origin,
				ax: [w, 0, 0],
				ay: [0, -h, 0],
				rows,
				cols: K,
				kind: 'ternary',
				painter: true,
				m0,
				x: { offset: 0, cap: true },
				gain: 1,
				pyramid: true,
				digits: true,
				lift: 1.5
			});
			this.p[key] = panel;
			this.mats.set(panel, { key, pre, m0 });
			return { w, h, panel };
		};
		// a matrix's running sums along its right (the finished sums once its sweep is done)
		const sums = (key: string, pre: string, m0: number, rows: number, origin: V3, h: number) => {
			const pa = this.pre[pre];
			this.p[key] = P.add({
				origin,
				ax: [0, -h, 0],
				ay: [1.4, 0, 0],
				rows: 1,
				cols: rows,
				kind: 'bars',
				a: { offset: pa.off + m0 * pa.nck, rs: 0, cs: pa.nck, aux: true },
				sweepStride: 1,
				sweep: pa.nck * 16 - 1,
				gain: 1,
				digits: true,
				lift: 1.2
			});
			return this.p[key];
		};
		// one row of a matrix spelled out: its weights, their products with the input, its running sum
		const inspector = (key: string, mat: string, pre: string, m0: number, origin: V3) => {
			const pa = this.pre[pre];
			const w = pa.K / U;
			const base: PanelDesc = {
				origin,
				ax: [w, 0, 0],
				ay: [0, -0.5, 0],
				rows: 1,
				cols: pa.K,
				kind: 'ternary',
				painter: true,
				gain: 1,
				digits: true,
				lift: 1.5
			};
			const wts = P.add({ ...base });
			const prod = P.add({
				...base,
				origin: [origin[0], origin[1] - 0.8, 0],
				x: { offset: 0, cap: true }
			});
			const run = P.add({
				origin: [origin[0], origin[1] - 1.6, 0],
				ax: [w, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: pa.nck,
				kind: 'bars',
				a: { offset: pa.off, rs: 0, cs: 1, aux: true },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.inspectors.set(key, { mat, pre, m0, wts, prod, run });
		};
		const inspectorLabels = (key: string, mat: string, iy: number, x0: number) => [
			{
				text: () => `row ${this.rowPick.get(key) ?? 0} of ${mat}: weights`,
				at: [x0 - 0.3, iy - 0.25, 0] as V3,
				size: 0.28,
				align: 'right' as const
			},
			{
				text: () => '× the input',
				at: [x0 - 0.3, iy - 1.05, 0] as V3,
				size: 0.28,
				align: 'right' as const
			},
			{
				text: () => 'running sum',
				at: [x0 - 0.3, iy - 2.4, 0] as V3,
				size: 0.28,
				align: 'right' as const
			}
		];

		let x = 0;
		const top = 0;
		let first = 0;
		const add = (s: Omit<Station, 'x' | 'first' | 'last'>, width: number) => {
			this.stations.push({ ...s, x, first, last: P.count });
			first = P.count;
			x += width + STATION_GAP;
		};
		const n = this.nums;
		const f = (v: number | undefined) => (v === undefined ? '' : fmt(v));
		const bn = () => this.block + 1;
		// normed, then set by the time step: five waveforms, top to bottom
		const setStation = (
			name: string,
			from: string,
			ln: number,
			sc: number,
			sh: number,
			to: string,
			caption: () => string
		) => {
			const x0 = x;
			const g = 2.3;
			wave(`${name}_x`, [x0, top, 0], 8, 1.6, D, cap(from));
			wave(`${name}_ln`, [x0, top - g, 0], 8, 1.6, D, aux(ln));
			wave(`${name}_sc`, [x0, top - 2 * g, 0], 8, 1.6, D, aux(sc));
			wave(`${name}_sh`, [x0, top - 3 * g, 0], 8, 1.6, D, aux(sh));
			wave(`${name}_y`, [x0, top - 4 * g, 0], 8, 1.6, D, cap(to));
			const l = (text: () => string, i: number) => ({
				text,
				at: [x0 - 0.4, top - i * g - 0.8, 0] as V3,
				size: 0.36,
				align: 'right' as const
			});
			add(
				{
					name,
					w: 8,
					h: 4 * g + 1.6,
					top,
					caption,
					labels: [
						l(() => `the ${this.what}`, 0),
						l(() => 'normed', 1),
						l(() => '× (1 + scale)', 2),
						l(() => '+ shift', 3),
						l(() => '=', 4)
					]
				},
				8
			);
		};
		// a matrix taking something back into the row through a gate: the matrix, its sums, then the addition
		const backStation = (
			name: string,
			mat: string,
			pre: string,
			inputName: string,
			inputLabel: string,
			from: string,
			gate: number,
			change: number,
			to: string,
			caption: () => string
		) => {
			const x0 = x;
			const pa = this.pre[pre];
			const w = pa.K / U;
			wave(`${mat}_in`, [x0, top + 1.9, 0], w, 1.5, pa.K, cap(inputName));
			const m = matrix(mat, pre, 0, D, [x0, top, 0]);
			sums(`${mat}_sum`, pre, 0, D, [x0 + w + 0.3, top, 0], m.h);
			const ay = top - m.h - 1.2;
			wave(`${mat}_a0`, [x0, ay, 0], 8, 1.2, D, cap(from));
			wave(`${mat}_a1`, [x0, ay - 1.8, 0], 8, 1.2, D, aux(gate));
			wave(`${mat}_a2`, [x0, ay - 3.6, 0], 8, 1.2, D, aux(change));
			wave(`${mat}_a3`, [x0, ay - 5.4, 0], 8, 1.2, D, cap(to));
			const iy = ay - 7.8;
			inspector(mat, mat, pre, 0, [x0, iy, 0]);
			const r = (text: () => string, y: number) => ({
				text,
				at: [x0 - 0.3, y, 0] as V3,
				size: 0.32,
				align: 'right' as const
			});
			add(
				{
					name,
					w: w + 1.7,
					h: -(iy - 3.4 - top - 3.4),
					top: top + 3.4,
					sweep: { mats: [m.panel], outs: [{ panel: this.p[`${mat}_sum`], K: pa.K }], K: pa.K },
					caption,
					labels: [
						{ text: () => inputLabel, at: [x0, top + 3.7, 0], size: 0.36, align: 'left' },
						r(() => `the ${this.what}`, ay - 0.6),
						r(() => 'gate (from the time step)', ay - 2.4),
						r(() => '+ gate × the sums', ay - 4.2),
						r(() => '=', ay - 6.0),
						...inspectorLabels(mat, mat, iy, x0)
					]
				},
				w + 1.7
			);
		};

		// 1. the row arrives
		{
			const x0 = x;
			wave('h', [x0, top, 0], 8, 2.6, D, cap('h_in'));
			this.pics.push({ which: 'before', at: [x0 + 9.5, top + 0.6, 0], size: 7 });
			add(
				{
					name: 'It arrives',
					w: 16.5,
					h: 7.6,
					top: top + 0.6,
					caption: () =>
						this.isPatch
							? `${cap1(this.rowName())} of the picture (one of 1,024, each 16 × 16 pixels) as it enters block ${bn()} at step ${this.step + 1} of ${this.run?.steps ?? 4}: 3,072 numbers. It began as noise and has been through the blocks before. On the right, the picture the painter had in mind before this block (read out with a lens fitted to each block), the patch ringed. A few of its numbers are far larger than the rest: they run off the top of the waveform and glow.`
							: `${this.rowName()} as the painter holds it entering block ${bn()} at step ${this.step + 1} of ${this.run?.steps ?? 4}: 3,072 numbers. The prompt's words go through the painter beside the picture: this one came from the reader (its layers 7, 14 and 21, through the adapter) and has been through the blocks before. A few of its numbers are far larger than the rest: they run off the top of the waveform and glow.`,
					labels: [
						{
							text: () => `${this.rowName()}, entering block ${bn()}`,
							at: [x0, top + 0.7, 0],
							size: 0.45,
							align: 'left',
							italic: true
						},
						{
							text: () => this.beforeText,
							at: [x0 + 9.5, top + 1.2, 0],
							size: 0.32,
							align: 'left'
						}
					]
				},
				16.5
			);
		}
		// 2. normed and set by the time step
		setStation(
			'Normed, set by the time',
			'h_in',
			A.ln1,
			A.sc1,
			A.sh1,
			'n1',
			() =>
				`First the ${this.what} is normed: less its mean (${f(n.mean1)}), divided by its spread (${f(n.std1)}). Then it is set by the time step: times (1 + scale), plus shift. Scale and shift come from the noise level alone (σ = ${f(n.sigma)} at this step), the same for every ${this.what}: it is how the painter knows how far along the painting is.`
		);
		// 3. q, k, v
		{
			const x0 = x;
			wave('qkv_in', [x0, top + 1.9, 0], 8, 1.5, D, cap('n1'));
			const keys = ['q', 'k', 'v'];
			let y = top;
			const panels: number[] = [];
			const ys: number[] = [];
			keys.forEach((k, i) => {
				const pre = dbl ? k : 'fused';
				const m0 = dbl ? 0 : i * D;
				const m = matrix(k, pre, m0, D, [x0, y, 0]);
				sums(`${k}_sum`, pre, m0, D, [x0 + 8.3, y, 0], m.h);
				panels.push(m.panel);
				ys.push(y);
				y -= m.h + 0.5;
			});
			const iy = y - 0.9;
			inspector('qkv', 'q', dbl ? 'q' : 'fused', 0, [x0, iy, 0]);
			add(
				{
					name: 'Three projections',
					w: 9.7,
					h: -(iy - 3.4 - top - 3.4),
					top: top + 3.4,
					sweep: {
						mats: panels,
						outs: keys.map((k) => ({ panel: this.p[`${k}_sum`], K: D })),
						K: D
					},
					caption: () =>
						(dbl
							? `Three matrices of 3,072 × 3,072 read it: q, k and v, the ${this.isPatch ? "picture's" : "words'"} own (in the first five blocks words and patches have separate weights). `
							: `In the twenty single blocks words and patches share their weights, and one matrix of 27,648 rows reads the ${this.what} once: its first 9,216 rows are q, k and v (here), the rest the MLP's gate and up (later). `) +
						`Each cell is a weight times the input above it; each row adds up into the waveform on its right. No multiplications: each weight adds its input (+1, amber), subtracts it (−1, blue) or skips it (0), and one scale per 128 weights sets the size. Below, row ${this.rowPick.get('qkv') ?? 0} of q spelled out, ${f(n.qkvRow)} at the end. Click any row of q to spell it out.`,
					labels: [
						{
							text: () => `the ${this.what}, normed and set`,
							at: [x0, top + 3.7, 0],
							size: 0.36,
							align: 'left'
						},
						...keys.map((k, i) => ({
							text: () => k,
							at: [x0 - 0.35, ys[i] - 4, 0] as V3,
							size: 0.6,
							align: 'right' as const,
							italic: true
						})),
						...inspectorLabels('qkv', 'q', iy, x0)
					]
				},
				9.7
			);
		}
		// 4. turned by position
		{
			const x0 = x;
			const dial = (key: string, y: number, a: number, b: number) =>
				(this.p[key] = P.add({
					origin: [x0, y, 0],
					ax: [16, 0, 0],
					ay: [0, -6, 0],
					rows: H,
					cols: 64,
					kind: 'dial',
					a: { offset: a, rs: HD, cs: 1, aux: true },
					b: { offset: b, rs: HD, cs: 1, aux: true },
					gain: 1
				}));
			dial('dial_q', top, A.qn, A.qr);
			const ky = top - 7.4;
			dial('dial_k', ky, A.kn, A.kr);
			const group = (text: string, i: number) => ({
				text: () => text,
				at: [x0 + (i + 0.5) * 4, top + 0.45, 0] as V3,
				size: 0.3,
				italic: true
			});
			add(
				{
					name: 'Turned by position',
					w: 16,
					h: 14.4,
					top: top + 1.4,
					caption: () =>
						`q and k split into 24 heads of 128 numbers, each normed on its own (with 128 learned gains), then turned: each head's 64 pairs are hands on dials. The pairs come in four groups of 16. The first never turn; the second turn with a patch's row on the picture, the third with its column; the last with a word's place in the prompt. So ${this.isPatch ? `this patch turns only in the middle two groups (row ${Math.floor(this.patch / GRID) + 1}, column ${(this.patch % GRID) + 1})` : `this word turns only in the last group (it is token ${this.row} of the prompt)`}: that is how attention can tell how far apart two patches are. Faint: before; bright: after.`,
					labels: [
						{
							text: () => 'q: 24 heads × 64 pairs',
							at: [x0, top + 1.2, 0],
							size: 0.36,
							align: 'left'
						},
						group('never turn', 0),
						group('the row', 1),
						group('the column', 2),
						group('the place in the prompt', 3),
						{ text: () => 'k', at: [x0, ky + 0.5, 0], size: 0.36, align: 'left' }
					]
				},
				16
			);
		}
		// 5. where it looks
		{
			const x0 = x;
			this.textAt = [x0, top, 0];
			this.p.shares_t = P.add({
				origin: [x0, top, 0],
				ax: [1, 0, 0], // sized to the prompt when it is read
				ay: [0, -6, 0],
				rows: H,
				cols: 1,
				kind: 'dense',
				a: { offset: 0, rs: NJ, cs: 1, cap: true },
				gain: 1,
				digits: true
			});
			this.p.shares_pad = P.add({
				origin: [x0 + 2, top, 0],
				ax: [1, 0, 0],
				ay: [0, -6, 0],
				rows: H,
				cols: 1,
				kind: 'dense',
				a: { offset: 0, rs: NJ, cs: 1, cap: true },
				gain: 1,
				grid: false
			});
			const my = top - 8.2;
			for (let h = 0; h < H; h++) {
				const at: V3 = [x0 + (h % 6) * 3.6, my - Math.floor(h / 6) * 3.9, 0];
				this.mapAt.push(at);
				this.mapSize.push(3.2);
				this.p[`map${h}`] = P.add({
					origin: at,
					ax: [3.2, 0, 0],
					ay: [0, -3.2, 0],
					rows: GRID,
					cols: GRID,
					kind: 'dense',
					a: { offset: 0, rs: GRID, cs: 1, cap: true },
					gain: 1,
					grid: false,
					lift: 1.4
				});
			}
			const rx = x0 + 22.6;
			this.mapAt.push([rx, my, 0]);
			this.mapSize.push(8);
			this.p.map = P.add({
				origin: [rx, my, 0],
				ax: [8, 0, 0],
				ay: [0, -8, 0],
				rows: GRID,
				cols: GRID,
				kind: 'dense',
				a: { offset: A.map, rs: GRID, cs: 1, aux: true },
				gain: 1,
				grid: false,
				lift: 1.4
			});
			const dy = my - 9.6;
			this.p.dot = P.add({
				origin: [rx, dy, 0],
				ax: [8, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: HD,
				kind: 'bars',
				a: { offset: 0, rs: 0, cs: 1, cap: true },
				b: { offset: 0, rs: 0, cs: 1, cap: true },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			this.p.dotsum = P.add({
				origin: [rx, dy - 2.3, 0],
				ax: [8, 0, 0],
				ay: [0, -1.6, 0],
				rows: 1,
				cols: HD,
				kind: 'bars',
				a: { offset: A.sumd, rs: 0, cs: 1, aux: true },
				gain: 1,
				digits: true,
				lift: 1.2
			});
			add(
				{
					name: 'Where it looks',
					w: 30.6,
					h: -(my - 4 * 3.9 - top) + 1,
					top: top + 1,
					caption: () =>
						`Each of the 24 heads compares this ${this.what}'s q with the k of all 1,536 rows, the prompt's 512 (its tokens, then padding) and the picture's 1,024, and softmax turns the scores into shares that add up to 1. This is the only place in the block where words and picture meet: ${pc(n.onPicture)} of this ${this.what}'s attention goes to the picture, ${pc(n.onWords)} to the prompt's words. Top: the shares for the prompt, one row per head. Below: each head's shares over the picture, 32 × 32 (each at its own brightness); right, the heads' mean. Under it, head ${this.head + 1} and ${this.rowName(this.other)}: q × k pair by pair and the running sum, ${f(n.dot)} (÷ √128 = ${f(n.score)}; share ${f(n.share)}). Click a word, or a patch on a map, to spell it out.`,
					labels: [
						{
							text: () => 'the prompt (one row per head)',
							at: [x0, top + 0.5, 0],
							size: 0.36,
							align: 'left'
						},
						{
							text: () => `padding (${NT - this.textCols} rows)`,
							at: this.padAt,
							size: 0.3,
							align: 'left',
							italic: true
						},
						{
							text: () => 'the picture, head by head',
							at: [x0, my + 0.55, 0],
							size: 0.36,
							align: 'left'
						},
						{ text: () => "the heads' mean", at: [rx, my + 0.55, 0], size: 0.36, align: 'left' },
						{
							text: () =>
								`head ${this.head + 1} × ${this.rowName(this.other)}: q × k, pair by pair`,
							at: [rx, dy + 0.5, 0],
							size: 0.28,
							align: 'left'
						},
						{ text: () => 'running sum', at: [rx, dy - 1.8, 0], size: 0.28, align: 'left' }
					]
				},
				30.6
			);
		}
		// 6. mixing the values
		{
			const x0 = x;
			this.p.heads = P.add({
				origin: [x0, top, 0],
				ax: [8, 0, 0],
				ay: [0, -6, 0],
				rows: H,
				cols: HD,
				kind: 'dense',
				a: { offset: 0, rs: HD, cs: 1, cap: true },
				gain: 1,
				digits: true
			});
			wave('head', [x0, top - 7.6, 0], 8, 1.6, HD, () => ({
				offset: this.capAt(dbl ? 'o' : 'cat') + this.head * HD,
				cap: true
			}));
			wave('vother', [x0, top - 10.6, 0], 8, 1.6, HD, () => ({
				offset: dbl
					? this.capAt('v', this.other) + this.head * HD
					: this.capAt('p', this.other) + 2 * D + this.head * HD,
				cap: true
			}));
			wave('term', [x0, top - 13.2, 0], 8, 1.6, HD, aux(A.term));
			add(
				{
					name: 'Mixing the values',
					w: 8,
					h: 15.8,
					top: top + 0.8,
					caption: () =>
						`A head's output is a sum over all 1,536 rows: each row's value (v, 128 numbers) times its share. Top, the 24 heads' outputs, one row each: 3,072 numbers, attention's result. Below, head ${this.head + 1}'s, and one of the 1,536 terms that make it: the value of ${this.rowName(this.other)} times its share, ${f(n.share)}.`,
					labels: [
						{
							text: () => "the 24 heads' outputs",
							at: [x0, top + 0.5, 0],
							size: 0.34,
							align: 'left'
						},
						{
							text: () => `head ${this.head + 1}'s output: the sum of 1,536 terms`,
							at: [x0, top - 7.1, 0],
							size: 0.3,
							align: 'left'
						},
						{
							text: () => `one term: the value of ${this.rowName(this.other)}`,
							at: [x0, top - 10.1, 0],
							size: 0.3,
							align: 'left'
						},
						{
							text: () => `× its share (${f(n.share)})`,
							at: [x0, top - 12.7, 0],
							size: 0.3,
							align: 'left'
						}
					]
				},
				8
			);
		}
		if (dbl) {
			// 7. back into the row
			backStation(
				'Back into the row',
				'o',
				'o',
				'o',
				"the heads' outputs",
				'h_in',
				A.g1,
				A.dA,
				'h_mid',
				() =>
					`The output projection (3,072 × 3,072, the ${this.isPatch ? "picture's" : "words'"}) turns the heads' outputs into 3,072 changes. Before they are added to the ${this.what}, each is multiplied by a gate that, like the scale and shift, comes from the time step alone. Below: the ${this.what}, the gate, what is added, and the result. Row ${this.rowPick.get('o') ?? 0} spelled out at the bottom.`
			);
			// 8. normed again
			setStation(
				'Normed and set again',
				'h_mid',
				A.ln2,
				A.sc2,
				A.sh2,
				'n2',
				() =>
					`Normed again (mean ${f(n.mean2)}, spread ${f(n.std2)}) and set by the time step with a second scale and shift, before the neurons.`
			);
		}
		// gate and up
		{
			const x0 = x;
			const pre = dbl ? 'in' : 'fused';
			const m0 = dbl ? 0 : 3 * D;
			const inName = dbl ? 'n2' : 'n1';
			wave('g_in', [x0, top + 1.9, 0], 8, 1.5, D, cap(inName));
			const g = matrix('gate', pre, m0, MLP, [x0, top, 0]);
			sums('gate_sum', pre, m0, MLP, [x0 + 8.3, top, 0], g.h);
			const ux = x0 + 11;
			wave('u_in', [ux, top + 1.9, 0], 8, 1.5, D, cap(inName));
			const u = matrix('up', pre, m0 + MLP, MLP, [ux, top, 0]);
			sums('up_sum', pre, m0 + MLP, MLP, [ux + 8.3, top, 0], u.h);
			add(
				{
					name: 'Gate and up',
					w: 20.7,
					h: g.h + 3.4,
					top: top + 3.4,
					sweep: {
						mats: [g.panel, u.panel],
						outs: [
							{ panel: this.p.gate_sum, K: D },
							{ panel: this.p.up_sum, K: D }
						],
						K: D
					},
					caption: () =>
						dbl
							? `The MLP: a matrix of 18,432 × 3,072 reads the ${this.what} as it was set again. Its first 9,216 rows are the gates of 9,216 neurons (how much each opens), the rest what each lets through (up).`
							: `The rest of the one matrix: 9,216 rows of gate (how much each of 9,216 neurons opens) and 9,216 of up (what each lets through), reading the same normed ${this.what} as q, k and v.`,
					labels: [
						{ text: () => 'gate', at: [x0, top + 3.7, 0], size: 0.5, align: 'left', italic: true },
						{ text: () => 'up', at: [ux, top + 3.7, 0], size: 0.5, align: 'left', italic: true }
					]
				},
				20.7
			);
		}
		// the neurons
		{
			const x0 = x;
			const g = 2.4;
			const wlen = 24;
			const pb = dbl ? 0 : 3 * D;
			wave('ne_g', [x0, top, 0], wlen, 1.6, MLP, cap('p', pb));
			wave('ne_s', [x0, top - g, 0], wlen, 1.6, MLP, aux(A.silu));
			wave('ne_u', [x0, top - 2 * g, 0], wlen, 1.6, MLP, cap('p', pb + MLP));
			wave('ne_a', [x0, top - 3 * g, 0], wlen, 1.6, MLP, cap('cat', dbl ? 0 : D));
			const hx = x0 + wlen + 2;
			this.p.hist = P.add({
				origin: [hx, top, 0],
				ax: [7.5, 0, 0],
				ay: [0, -7.5, 0],
				rows: HB,
				cols: HB,
				kind: 'dense',
				a: { offset: A.hist, rs: HB, cs: 1, aux: true },
				gain: 1,
				grid: false,
				lift: 2
			});
			const l = (text: string, i: number) => ({
				text: () => text,
				at: [x0 - 0.4, top - i * g - 0.8, 0] as V3,
				size: 0.36,
				align: 'right' as const
			});
			add(
				{
					name: 'The neurons',
					w: wlen + 9.5,
					h: 3 * g + 1.6,
					top,
					caption: () =>
						`The 9,216 neurons: each gate goes through SiLU (right: every neuron placed by its gate and what SiLU makes of it, so together they trace the curve), then is multiplied by up. ${n.fired ?? 0} of 9,216 come out larger than a tenth of the largest.`,
					labels: [
						l('gate', 0),
						l('SiLU(gate)', 1),
						l('× up', 2),
						l('= the neurons', 3),
						{
							text: () => 'SiLU(gate) against gate',
							at: [hx, top + 0.5, 0],
							size: 0.34,
							align: 'left'
						},
						{ text: () => `${f(-n.hr)}`, at: [hx, top - 7.9, 0], size: 0.28, align: 'left' },
						{ text: () => `${f(n.hr)}`, at: [hx + 7.5, top - 7.9, 0], size: 0.28, align: 'right' }
					]
				},
				wlen + 9.5
			);
		}
		if (dbl)
			backStation(
				'Down, added',
				'down',
				'out',
				'cat',
				'the neurons',
				'h_mid',
				A.g2,
				A.dM,
				'h_out',
				() =>
					`Down (3,072 × 9,216) reads the neurons and writes 3,072 changes, gated by the time step and added. The ${this.what} leaves block ${bn()}. Row ${this.rowPick.get('down') ?? 0} spelled out at the bottom: 9,216 weights.`
			);
		else
			backStation(
				'One matrix back',
				'out',
				'out',
				'cat',
				"attention's result and the neurons, side by side",
				'h_in',
				A.g1,
				A.dA,
				'h_out',
				() =>
					`One matrix (3,072 × 12,288) reads attention's result and the neurons side by side and writes 3,072 changes, gated by the time step and added: the ${this.what} leaves block ${bn()}. Row ${this.rowPick.get('out') ?? 0} spelled out at the bottom: 12,288 weights.`
			);
		// the picture, and the row through all the blocks
		{
			const x0 = x;
			const step = 0.42;
			this.pics.push({ which: 'after', at: [x0, top, 0], size: 12 });
			const sx = x0 + 14.5;
			this.p.stack = P.add({
				origin: [sx, top, 0],
				ax: [8, 0, 0],
				ay: [0, -26 * step, 0],
				rows: 26,
				cols: D,
				kind: 'dense',
				a: { offset: A.stack, rs: D, cs: 1, aux: true },
				gain: 1,
				grid: false,
				lift: 1.2
			});
			add(
				{
					name: 'The picture',
					w: 22.5,
					h: 12,
					top,
					caption: () =>
						`After block ${bn()}, the picture the painter has in mind (what it would paint if it stopped here, read out with a lens fitted to each block)${this.isPatch ? ', the patch ringed' : ''}. Right: the ${this.what} entering the first block and after each of the 25, top to bottom (each row the same 3,072 numbers). Every block does what you just watched, with its own weights; ${this.isPatch ? 'every patch and every word at once, all 1,536 rows' : 'every word and every patch at once, all 1,536 rows'}.`,
					labels: [
						{
							text: () => `after block ${bn()}, step ${this.step + 1}`,
							at: [x0, top + 0.5, 0],
							size: 0.36,
							align: 'left'
						},
						{
							text: () => 'entering',
							at: [sx - 0.3, top - step * 0.5, 0],
							size: 0.28,
							align: 'right'
						},
						{
							text: () => `after block ${bn()} →`,
							at: [sx - 0.3, top - step * (this.block + 1.5), 0],
							size: 0.3,
							align: 'right'
						},
						{
							text: () => 'after 25',
							at: [sx - 0.3, top - step * 25.5, 0],
							size: 0.28,
							align: 'right'
						}
					]
				},
				22.5
			);
		}
		P.allocate();
	}

	private textCols = 1;
	private padX = 0;
	private padAt: V3 = [0, 0, 0];
	private finCount = 0;
	/** Every matrix's finished sums for this row (read back after the running sums). */
	private fin = new Float32Array(0);

	/** The finished sums of rows [m0, m0 + n) of a matrix. */
	private finals(pre: string, m0 = 0, n = this.pre[pre].M - m0) {
		const o = this.pre[pre].fin + m0;
		return this.fin.subarray(o, o + n);
	}

	private async readAux(offset: number, count: number) {
		const dev = this.stage.device;
		const buf = dev.createBuffer({
			size: count * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = dev.createCommandEncoder();
		enc.copyBufferToBuffer(this.panels.aux, offset * 4, buf, 0, count * 4);
		dev.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const v = new Float32Array(buf.getMappedRange().slice(0));
		buf.destroy();
		return v;
	}

	private get beforeText() {
		const b = this.block,
			s = this.step;
		if (b > 0) return `the picture after block ${b}`;
		if (s > 0) return `the picture after step ${s}`;
		return 'before the first block: noise';
	}

	private beforeTexture(): GPUTexture | null {
		const r = this.run,
			b = this.block,
			s = this.step;
		if (!r) return null;
		if (b > 0) return r.pictures[s]?.[b - 1] ?? null;
		if (s > 0) return r.pictures[s - 1]?.[24] ?? null;
		return null;
	}

	// ---- choosing

	/** A new painting (its capture is where the numbers come from). */
	setRun(run: PaintingRun) {
		this.run = run;
		this.panels.setCapture(run.cap.buffer);
		this.planeOf.clear();
		this.planes.clear();
		this.cpu = undefined;
		if (!this.fullRows.includes(this.row)) this.row = NT + 16 * GRID + 16;
		if (!this.fullRows.includes(this.row)) this.row = this.fullRows[0];
		void this.load();
	}

	async choose(o: { row?: number; block?: number }) {
		if (o.row !== undefined && this.fullRows.includes(o.row)) this.row = o.row;
		if (o.block !== undefined) this.block = Math.max(0, Math.min(24, o.block));
		await this.load();
	}

	/** The followed patch nearest a patch of the picture. */
	nearestFollowed(p: number) {
		const pats = this.run?.opts.patches ?? [];
		let best = pats[0] ?? p,
			bd = Infinity;
		for (const q of pats) {
			const d = (Math.floor(q / GRID) - Math.floor(p / GRID)) ** 2 + ((q % GRID) - (p % GRID)) ** 2;
			if (d < bd) {
				bd = d;
				best = q;
			}
		}
		return best;
	}

	// ---- reading

	/** Everything for the chosen row, block and step: read back, derive what the painter does not keep, point. */
	private async load() {
		const run = this.run;
		if (!run) return;
		const id = ++this.loadId;
		const s = this.step,
			b = this.block,
			r = this.row;
		if (!run.has(s, b)) {
			this.waiting = true;
			this.cpu = undefined;
			return;
		}
		this.waiting = false;
		this.busy = 'Collecting every number of this block';
		const kind: Kind = b < 5 ? 'double' : 'single';
		if (kind !== this.kind) {
			this.build(kind);
			this.goTo(Math.min(this.station, this.stations.length - 1));
		}
		const dbl = kind === 'double';
		const cap = run.cap;
		const sizes: Record<string, number> = {
			h_in: D,
			n1: D,
			q: D,
			k: D,
			v: D,
			qr: D,
			kr: D,
			attn: H * NJ,
			o: D,
			h_mid: D,
			n2: D,
			p: dbl ? 2 * MLP : 3 * D + 2 * MLP,
			cat: dbl ? MLP : D + MLP,
			h_out: D
		};
		const names = dbl
			? ['h_in', 'n1', 'q', 'k', 'v', 'qr', 'kr', 'attn', 'o', 'h_mid', 'n2', 'p', 'cat', 'h_out']
			: ['h_in', 'n1', 'p', 'qr', 'kr', 'attn', 'cat', 'h_out'];
		const regs = names.map((nm) => ({ offset: cap.at(s, b, r, nm)!, count: sizes[nm] }));
		// the row through all the blocks
		const stackRegs = [
			{ offset: cap.at(s, 0, r, 'h_in')!, count: D },
			...Array.from({ length: 25 }, (_, bb) => ({ offset: cap.at(s, bb, r, 'h_out')!, count: D }))
		];
		// the other kept rows' turned keys and values (for the dot product and the term)
		const others = this.fullRows.filter((x) => x !== r);
		const otherRegs = others.flatMap((o) => [
			{ offset: cap.at(s, b, o, 'kr')!, count: D },
			dbl
				? { offset: cap.at(s, b, o, 'v')!, count: D }
				: { offset: cap.at(s, b, o, 'p')! + 2 * D, count: D }
		]);
		if ([...regs, ...stackRegs, ...otherRegs].some((x) => x.offset === undefined)) {
			this.busy = 'This row is not kept at this step';
			return;
		}
		const pt = this.painter;
		if (!this.mod.has(s))
			this.mod.set(s, await pt.readFloats('arena', pt.at.mod + s * 17 * D, 17 * D));
		const W = pt.weights;
		const [gq, gk] = await Promise.all([
			pt.readFloats('dense', W.dense_at(this.normName('q')), HD),
			pt.readFloats('dense', W.dense_at(this.normName('k')), HD)
		]);
		const got = await run.read([...regs, ...stackRegs, ...otherRegs]);
		if (id !== this.loadId) return;
		const v: Record<string, Float32Array> = {};
		names.forEach((nm, i) => (v[nm] = got[i]));
		const stack = got.slice(names.length, names.length + 26);
		const oth = new Map<number, { kr: Float32Array; v: Float32Array }>();
		others.forEach((o, i) =>
			oth.set(o, {
				kr: got[names.length + 26 + 2 * i],
				v: got[names.length + 26 + 2 * i + 1]
			})
		);
		const mod = this.mod.get(s)!;
		const img = r >= NT;
		// rows of the step's modulation: double blocks (image, then text) shift, scale, gate x 2; single blocks x 1
		const M = (i: number) => {
			const base = dbl ? (img ? 0 : 6) + i : 12 + i;
			return mod.subarray(base * D, (base + 1) * D);
		};
		const aux = new Float32Array(A.prefix);
		const ln = (x: Float32Array) => {
			let mean = 0;
			for (const a of x) mean += a / x.length;
			let vr = 0;
			for (const a of x) vr += (a - mean) ** 2 / x.length;
			const std = Math.sqrt(vr + EPS);
			return { y: x.map((a) => (a - mean) / std), mean, std };
		};
		const l1 = ln(v.h_in);
		aux.set(l1.y, A.ln1);
		aux.set(
			M(1).map((a) => 1 + a),
			A.sc1
		);
		aux.set(M(0), A.sh1);
		// q and k per head, normed (before turning), pairs split for the dials
		const qsrc = dbl ? v.q : v.p.subarray(0, D),
			ksrc = dbl ? v.k : v.p.subarray(D, 2 * D);
		const split = (src: Float32Array, dst: number, gain?: Float32Array) => {
			for (let h = 0; h < H; h++) {
				const seg = src.subarray(h * HD, (h + 1) * HD);
				let ss = 0;
				for (const a of seg) ss += a * a;
				const inv = gain ? 1 / Math.sqrt(ss / HD + EPS) : 1;
				for (let p = 0; p < 64; p++) {
					aux[dst + h * HD + p] = seg[2 * p] * inv * (gain ? gain[2 * p] : 1);
					aux[dst + h * HD + 64 + p] = seg[2 * p + 1] * inv * (gain ? gain[2 * p + 1] : 1);
				}
			}
		};
		split(qsrc, A.qn, gq);
		split(v.qr, A.qr);
		split(ksrc, A.kn, gk);
		split(v.kr, A.kr);
		// where it looks
		const attn = v.attn;
		let onPic = 0,
			onWords = 0;
		for (let j = 0; j < 1024; j++) {
			let m = 0;
			for (let h = 0; h < H; h++) m += attn[h * NJ + NT + j] / H;
			aux[A.map + j] = m;
			onPic += m;
		}
		for (const w of run.words) for (let h = 0; h < H; h++) onWords += attn[h * NJ + w] / H;
		// the row spelled out: the kept row this one gives the largest share (and that head), unless one is chosen
		if (!others.includes(this.other)) {
			let best = -1;
			for (const o of others)
				for (let h = 0; h < H; h++)
					if (attn[h * NJ + o] > best) {
						best = attn[h * NJ + o];
						this.other = o;
						this.head = h;
					}
		}
		// the gates and what the block adds
		this.gateOf = dbl ? [M(2), M(5)] : [M(2)];
		if (dbl) {
			aux.set(M(2), A.g1);
			aux.set(
				v.h_mid.map((a, i) => a - v.h_in[i]),
				A.dA
			);
			const l2 = ln(v.h_mid);
			aux.set(l2.y, A.ln2);
			aux.set(
				M(4).map((a) => 1 + a),
				A.sc2
			);
			aux.set(M(3), A.sh2);
			aux.set(M(5), A.g2);
			aux.set(
				v.h_out.map((a, i) => a - v.h_mid[i]),
				A.dM
			);
			Object.assign(this.nums, { mean2: l2.mean, std2: l2.std });
		} else {
			aux.set(M(2), A.g1);
			aux.set(
				v.h_out.map((a, i) => a - v.h_in[i]),
				A.dA
			);
		}
		// the neurons
		const gate = dbl ? v.p.subarray(0, MLP) : v.p.subarray(3 * D, 3 * D + MLP);
		const act = dbl ? v.cat : v.cat.subarray(D);
		const silu = gate.map((g) => g / (1 + Math.exp(-g)));
		aux.set(silu, A.silu);
		let hr = 0;
		for (const g of gate) hr = Math.max(hr, Math.abs(g));
		hr = Math.max(hr, 1e-6);
		for (let i = 0; i < MLP; i++) {
			const bx = Math.min(HB - 1, Math.floor(((gate[i] / hr) * 0.5 + 0.5) * HB));
			const by = Math.min(HB - 1, Math.max(0, Math.floor((0.5 - (silu[i] / hr) * 0.5) * HB)));
			aux[A.hist + by * HB + bx] += 1;
		}
		for (let i = 0; i < HB * HB; i++) aux[A.hist + i] = Math.log1p(aux[A.hist + i]);
		let mact = 0;
		for (const a of act) mact = Math.max(mact, Math.abs(a));
		stack.forEach((x, i) => aux.set(x, A.stack + i * D));
		Object.assign(this.nums, {
			mean1: l1.mean,
			std1: l1.std,
			sigma: pt.sigmas[s],
			hr,
			onPicture: onPic,
			onWords,
			fired: act.reduce((c, a) => c + (Math.abs(a) > mact * 0.1 ? 1 : 0), 0)
		});
		this.cpu = { ...v, gate, act };
		this.othersKept = oth;
		this.stage.device.queue.writeBuffer(this.panels.aux, 0, aux);
		if (import.meta.env.DEV) this.check(v, l1.y, M);

		// point every panel at this row, block and step
		const P = this.panels;
		for (const [panel, src] of this.srcs) {
			const d = P.desc(panel);
			const x = src();
			P.set(panel, { a: { ...d.a!, offset: x.offset, aux: x.aux, cap: x.cap } });
		}
		for (const [panel, m] of this.mats) {
			const t = W.ternary(this.tensorName(m.pre));
			P.set(panel, {
				tensor: { ...t, meanScale: 0 },
				x: { offset: this.capAt(this.pre[m.pre].input), cap: true }
			});
		}
		const words = run.tokens.length;
		this.textCols = Math.min(NT, words);
		const tc = this.textCell;
		this.padX = this.textAt[0] + words * tc + 0.6;
		this.padAt.splice(0, 3, this.padX, this.textAt[1] + 0.5, 0);
		const ao = this.capAt('attn');
		P.set(this.p.shares_t, {
			cols: this.textCols,
			ax: [this.textCols * tc, 0, 0],
			a: { offset: ao, rs: NJ, cs: 1, cap: true }
		});
		const padW = Math.max(1, 22 - words * tc - 0.6);
		P.set(this.p.shares_pad, {
			origin: [this.padX, this.textAt[1], 0],
			cols: NT - this.textCols,
			ax: [padW, 0, 0],
			a: { offset: ao + this.textCols, rs: NJ, cs: 1, cap: true }
		});
		P.set(this.p.heads, { a: { offset: this.capAt(dbl ? 'o' : 'cat'), rs: HD, cs: 1, cap: true } });
		for (let h = 0; h < H; h++)
			P.set(this.p[`map${h}`], { a: { offset: ao + h * NJ + NT, rs: GRID, cs: 1, cap: true } });
		this.setDot(this.head, this.other);
		// running sums, pyramids, brightness
		const enc = this.stage.device.createCommandEncoder();
		this.runPrefix(enc);
		P.buildPyramids(enc);
		this.stage.device.queue.submit([enc.finish()]);
		const matPanels = [...this.mats.keys()];
		const [means, fin] = await Promise.all([
			P.meanAbs(matPanels),
			this.readAux(A.fin, this.finCount)
		]);
		if (id !== this.loadId) return;
		this.fin = fin;
		if (import.meta.env.DEV) {
			const q = this.finals(dbl ? 'q' : 'fused', 0, D);
			let e = 0;
			for (let i = 0; i < D; i++) e = Math.max(e, Math.abs(q[i] - (dbl ? v.q : v.p)[i]));
			console.debug(
				`block lab: q from the running sums against the painter's, largest difference ${e}`
			);
		}
		matPanels.forEach((p, j) => P.set(p, { gain: means[j] * 18 || 1 }));
		this.gains(aux, v, gate, act);
		for (const key of this.inspectors.keys()) if (!this.rowPick.has(key)) this.rowPick.set(key, -1);
		await this.pickRows();
		if (id !== this.loadId) return;
		this.busy = '';
	}

	private othersKept = new Map<number, { kr: Float32Array; v: Float32Array }>();

	/** In development: the numbers derived here against the painter's own (normed and set; turned). */
	private check(
		v: Record<string, Float32Array>,
		ln1: Float32Array,
		M: (i: number) => Float32Array
	) {
		let e = 0,
			m = 0;
		for (let i = 0; i < D; i++) {
			const y = ln1[i] * (1 + M(1)[i]) + M(0)[i];
			e = Math.max(e, Math.abs(y - v.n1[i]));
			m = Math.max(m, Math.abs(v.n1[i]));
		}
		console.debug(`block lab: normed and set, largest difference ${e} (largest value ${m})`);
	}

	/** Brightness of the waveforms and grids: each at its largest value. */
	private gains(
		aux: Float32Array,
		v: Record<string, Float32Array>,
		gate: Float32Array,
		act: Float32Array
	) {
		const mx = (a: ArrayLike<number>) => {
			let m = 0;
			for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
			return m || 1;
		};
		// the painter's rows carry a few numbers far larger than the rest: a waveform is scaled to the bulk (a little
		// above the largest of all but the top half percent), and the few beyond run off its top and glow
		const rb = (a: ArrayLike<number>) => {
			const s = Float32Array.from(a, Math.abs).sort();
			return s[Math.floor(s.length * 0.995)] * 1.4 || mx(a);
		};
		const ax = (o: number, n: number) => aux.subarray(o, o + n);
		const P = this.panels,
			p = this.p;
		const set = (key: string, g: number) => p[key] !== undefined && P.set(p[key], { gain: g });
		const dbl = this.kind === 'double';
		const sets = dbl
			? ['Normed, set by the time', 'Normed and set again']
			: ['Normed, set by the time'];
		sets.forEach((nm, i) => {
			const src = i === 0 ? v.h_in : v.h_mid;
			const y = i === 0 ? v.n1 : v.n2;
			set(`${nm}_x`, rb(src));
			set(`${nm}_ln`, rb(ax(i === 0 ? A.ln1 : A.ln2, D)));
			set(`${nm}_sc`, rb(ax(i === 0 ? A.sc1 : A.sc2, D)));
			set(`${nm}_sh`, rb(ax(i === 0 ? A.sh1 : A.sh2, D)));
			set(`${nm}_y`, rb(y));
		});
		set('h', rb(v.h_in));
		set('qkv_in', rb(v.n1));
		const q = dbl ? v.q : v.p.subarray(0, D),
			k = dbl ? v.k : v.p.subarray(D, 2 * D),
			vv = dbl ? v.v : v.p.subarray(2 * D, 3 * D);
		set('q_sum', rb(q));
		set('k_sum', rb(k));
		set('v_sum', rb(vv));
		set('dial_q', mx(ax(A.qr, D)) * 1.1);
		set('dial_k', mx(ax(A.kr, D)) * 1.1);
		// shares: the prompt's at the largest share on it, each head's map at its own largest
		const attn = v.attn;
		let mt = 0;
		for (let h = 0; h < H; h++) for (let j = 0; j < NT; j++) mt = Math.max(mt, attn[h * NJ + j]);
		set('shares_t', mt || 1);
		set('shares_pad', mt || 1);
		for (let h = 0; h < H; h++) set(`map${h}`, mx(attn.subarray(h * NJ + NT, (h + 1) * NJ)));
		set('map', mx(ax(A.map, 1024)));
		const heads = dbl ? v.o : v.cat.subarray(0, D);
		set('heads', rb(heads) * 0.7);
		const mid = dbl ? v.h_mid : v.h_out;
		if (dbl) {
			set('o_in', rb(v.o));
			set('o_sum', rb(this.finals('o')));
			set('o_a0', rb(v.h_mid));
			set('o_a1', rb(ax(A.g1, D)));
			set('o_a2', rb(v.h_mid));
			set('o_a3', rb(v.h_mid));
			set('down_in', rb(v.cat));
			set('down_sum', rb(this.finals('out')));
			set('down_a0', rb(v.h_out));
			set('down_a1', rb(ax(A.g2, D)));
			set('down_a2', rb(v.h_out));
			set('down_a3', rb(v.h_out));
		} else {
			set('out_in', rb(v.cat));
			set('out_sum', rb(this.finals('out')));
			set('out_a0', rb(mid));
			set('out_a1', rb(ax(A.g1, D)));
			set('out_a2', rb(mid));
			set('out_a3', rb(mid));
		}
		const up = dbl ? v.p.subarray(MLP, 2 * MLP) : v.p.subarray(3 * D + MLP);
		set('g_in', rb(dbl ? v.n2 : v.n1));
		set('u_in', rb(dbl ? v.n2 : v.n1));
		set('gate_sum', rb(gate));
		set('up_sum', rb(up));
		set('ne_g', rb(gate));
		set('ne_s', rb(gate));
		set('ne_u', rb(up));
		set('ne_a', rb(act));
		set('hist', Math.log1p(40));
		set('stack', rb(ax(A.stack, 26 * D)));
	}

	/** Head h's q against row o's k, pair by pair, and the term o adds to the head's output. */
	setDot(h: number, o: number) {
		const cpu = this.cpu,
			kept = this.othersKept.get(o);
		if (!cpu || !kept) return;
		this.head = h;
		this.other = o;
		const sum = new Float32Array(HD);
		let acc = 0,
			mx = 0,
			mp = 0;
		for (let i = 0; i < HD; i++) {
			const pr = cpu.qr[h * HD + i] * kept.kr[h * HD + i];
			acc += pr;
			sum[i] = acc;
			mx = Math.max(mx, Math.abs(acc));
			mp = Math.max(mp, Math.abs(pr));
		}
		const share = cpu.attn[h * NJ + o];
		const term = new Float32Array(HD);
		let mv = 0,
			mt = 0;
		for (let i = 0; i < HD; i++) {
			term[i] = share * kept.v[h * HD + i];
			mv = Math.max(mv, Math.abs(kept.v[h * HD + i]));
			mt = Math.max(mt, Math.abs(term[i]));
		}
		Object.assign(this.nums, { dot: acc, score: acc / Math.sqrt(HD), share });
		const dev = this.stage.device;
		dev.queue.writeBuffer(this.panels.aux, A.sumd * 4, sum);
		dev.queue.writeBuffer(this.panels.aux, A.term * 4, term);
		const P = this.panels;
		P.set(this.p.dot, {
			a: { offset: this.capAt('qr') + h * HD, rs: 0, cs: 1, cap: true },
			b: { offset: this.capAt('kr', o) + h * HD, rs: 0, cs: 1, cap: true },
			gain: mp || 1
		});
		P.set(this.p.dotsum, { gain: mx || 1 });
		const dbl = this.kind === 'double';
		const out = dbl ? cpu.o : cpu.cat.subarray(0, D);
		let mo = 0;
		for (let i = 0; i < HD; i++) mo = Math.max(mo, Math.abs(out[h * HD + i]));
		P.set(this.p.head, {
			a: { offset: this.capAt(dbl ? 'o' : 'cat') + h * HD, rs: 0, cs: 1, cap: true },
			gain: mo || 1
		});
		P.set(this.p.vother, {
			a: {
				offset: dbl ? this.capAt('v', o) + h * HD : this.capAt('p', o) + 2 * D + h * HD,
				rs: 0,
				cs: 1,
				cap: true
			},
			gain: mv || 1
		});
		P.set(this.p.term, { gain: Math.max(mt, mo) || 1 });
	}

	/** Record the running sums of the block's matrices for this row. */
	private runPrefix(enc: GPUCommandEncoder) {
		const dev = this.stage.device;
		const W = this.painter.weights;
		const keys = Object.keys(this.pre);
		const data = new Uint32Array((keys.length * 256) / 4);
		keys.forEach((k, i) => {
			const pa = this.pre[k];
			const t = W.ternary(this.tensorName(k));
			data.set(
				[t.codes, t.scales, pa.K, pa.M, this.capAt(pa.input), pa.off, pa.nck, A.fin + pa.fin],
				i * 64
			);
		});
		dev.queue.writeBuffer(this.prefixUniforms, 0, data);
		keys.forEach((k, i) => {
			const pa = this.pre[k];
			const bg = dev.createBindGroup({
				layout: this.prefixPipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.prefixUniforms, offset: i * 256, size: 32 } },
					{ binding: 1, resource: { buffer: W.codes } },
					{ binding: 2, resource: { buffer: W.scales } },
					{ binding: 3, resource: { buffer: this.run!.cap.buffer } },
					{ binding: 4, resource: { buffer: this.panels.aux } }
				]
			});
			const pass = enc.beginComputePass();
			pass.setPipeline(this.prefixPipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(Math.ceil(pa.M / 64));
			pass.end();
		});
	}

	/** Point the inspectors at their rows (by default the row that adds the most). */
	private async pickRows() {
		const cpu = this.cpu;
		if (!cpu) return;
		// a row of the output projections by what it adds (after the gate), q's by its size
		const add = (pre: string, gate: number) => {
			const f = this.finals(pre);
			const g = new Float32Array(D);
			for (let i = 0; i < D; i++) g[i] = f[i] * (this.gateOf[gate]?.[i] ?? 1);
			return g;
		};
		const outs: Record<string, Float32Array> = {
			qkv: this.finals(this.kind === 'double' ? 'q' : 'fused', 0, D),
			o: this.pre.o ? add('o', 0) : new Float32Array(0),
			down: this.kind === 'double' ? add('out', 1) : new Float32Array(0),
			out: this.kind === 'single' ? add('out', 0) : new Float32Array(0)
		};
		const P = this.panels,
			W = this.painter.weights;
		for (const [key, ins] of this.inspectors) {
			const o = outs[key];
			let row = this.rowPick.get(key) ?? -1;
			if (row < 0 && o) {
				row = 0;
				for (let i = 1; i < o.length; i++) if (Math.abs(o[i]) > Math.abs(o[row])) row = i;
			}
			row = Math.max(0, row);
			this.rowPick.set(key, row);
			const pa = this.pre[ins.pre];
			const t = W.ternary(this.tensorName(ins.pre));
			const tensor = { ...t, meanScale: 0 };
			const m0 = ins.m0 + row;
			const scales = await this.painter.readFloats(
				'scales',
				t.scales + m0 * (pa.K / 128),
				pa.K / 128
			);
			let ms = 0;
			for (const a of scales) ms += Math.abs(a) / scales.length;
			const mat = this.p[ins.mat];
			const g = (P.desc(mat).gain || 1) / 18;
			const x = this.capAt(pa.input);
			P.set(ins.wts, { tensor, m0, gain: ms * 1.2 || 1 });
			P.set(ins.prod, { tensor, m0, x: { offset: x, cap: true }, gain: g * 6 });
			// the running sum: at the brightness of the largest it reaches
			const rs = this.fin.length ? this.finals(ins.pre, m0, 1)[0] : 0;
			const run = await this.readAux(pa.off + m0 * pa.nck, pa.nck);
			let peak = 1e-6;
			for (const a of run) peak = Math.max(peak, Math.abs(a));
			P.set(ins.run, {
				a: { offset: pa.off + m0 * pa.nck, rs: 0, cs: 1, aux: true },
				gain: peak * 1.1
			});
			if (key === 'qkv') this.nums.qkvRow = rs;
		}
	}
	/** The gates of the step (first, second), for what a row of an output projection adds. */
	private gateOf: Float32Array[] = [];

	// ---- time and camera

	goTo(i: number) {
		this.station = Math.max(0, Math.min(this.stations.length - 1, i));
		this.t = 0;
		const s = this.stations[this.station];
		const c = this.stage.gpu.canvas;
		const aspect = c.clientWidth / Math.max(1, c.clientHeight);
		const tn = Math.tan((38 * Math.PI) / 360);
		// fit it in the band between the controls above and the caption below (under half the height, a little below
		// the middle, and 70% of the width)
		const w = s.w + 4,
			h = s.h + 1.5;
		const dist = Math.max(w / (2 * tn * aspect * 0.7), h / (2 * tn * 0.46));
		const shift = dist * tn * 0.075;
		this.stage.flyTo({
			target: [s.x + s.w / 2 - 1, s.top - s.h / 2 + shift, 0],
			dist,
			yaw: 0.05,
			pitch: 0.02
		});
	}

	next() {
		if (this.station < this.stations.length - 1) this.goTo(this.station + 1);
		else this.playing = false;
	}

	prev() {
		this.goTo(this.station - 1);
	}

	private stackKey = '';
	/** Re-read the row through all the blocks as the painting reaches them. */
	private async refreshStack() {
		const run = this.run;
		if (!run || !this.cpu || this.waiting) return;
		const key = `${run.at.step}|${run.at.block}|${run.done}|${this.row}|${this.loadId}`;
		if (key === this.stackKey) return;
		this.stackKey = key;
		const s = this.step,
			r = this.row,
			cap = run.cap;
		const regs = [
			{ offset: cap.at(s, 0, r, 'h_in')!, count: D },
			...Array.from({ length: 25 }, (_, b) => ({
				offset: cap.at(s, b, r, 'h_out')!,
				count: run.has(s, b) ? D : 0
			}))
		].filter((x) => x.count > 0);
		const got = await run.read(regs);
		if (key !== this.stackKey) return;
		const all = new Float32Array(26 * D);
		got.forEach((x, i) => all.set(x, i * D));
		this.stage.device.queue.writeBuffer(this.panels.aux, A.stack * 4, all);
		const sorted = all
			.subarray(0, got.length * D)
			.map(Math.abs)
			.sort();
		const g = sorted[Math.floor(sorted.length * 0.995)] * 1.4;
		if (g > 0) this.panels.set(this.p.stack, { gain: g });
	}

	update(dt: number) {
		this.t += dt;
		void this.refreshStack();
		if (this.waiting && this.run?.has(this.step, this.block)) void this.load();
		if (this.waiting && this.run) {
			const at = this.run.at;
			this.busy =
				at.block < 0 && at.step === 0
					? 'The painter is reading the prompt'
					: `The painter is at step ${at.step + 1}, block ${Math.min(25, at.block + 1)}; this is block ${this.block + 1} of step ${this.step + 1}`;
		}
		if (this.playing && !this.waiting && this.t > this.duration(this.station)) this.next();
		const P = this.panels;
		const lit = !this.waiting && !!this.cpu;
		this.stations.forEach((s, i) => {
			const a = !lit ? 0.06 : i === this.station ? 1 : 0.14;
			for (let p = s.first; p < s.last; p++) P.set(p, { alpha: a });
		});
		this.stations.forEach((s, i) => {
			if (!s.sweep) return;
			const K = s.sweep.K;
			const here = i === this.station;
			const f = here ? Math.min(1, Math.max(0, (this.t - 1.2) / 5.5)) : 1;
			const sw = f * K;
			for (const m of s.sweep.mats) P.set(m, { sweep: f < 1 ? sw : null });
			for (const o of s.sweep.outs) P.set(o.panel, { sweep: Math.min(o.K - 1, sw) });
		});
		// the pictures before and after the block
		for (const plane of this.planeOf.values())
			this.planes.place(plane, [0, 0, 0], [1, 0, 0], [0, -1, 0], 0);
		for (const pic of this.pics) {
			const tex =
				pic.which === 'before'
					? this.beforeTexture()
					: (this.run?.pictures[this.step]?.[this.block] ?? null);
			if (!tex || !lit) continue;
			let plane = this.planeOf.get(tex);
			if (plane === undefined) {
				plane = this.planes.add(tex);
				this.planeOf.set(tex, plane);
			}
			const st = this.stations.find((s) => pic.at[0] >= s.x && pic.at[0] <= s.x + s.w);
			const a = st && this.stations.indexOf(st) === this.station ? 1 : 0.25;
			this.planes.place(plane, pic.at, [pic.size, 0, 0], [0, -pic.size, 0], 1, a);
		}
	}

	private duration(i: number) {
		return this.stations[i].sweep ? 10 : 8;
	}

	draw(pass: GPURenderPassEncoder) {
		this.planes.draw(pass, this.stage.eye);
		this.panels.draw(pass);
	}

	labels(): LayerLabel[] {
		if (!this.cpu || this.waiting) return [];
		const s = this.stage;
		const out: LayerLabel[] = [];
		for (const st of this.stations) {
			for (const l of st.labels) {
				const q = s.project(l.at);
				if (!q.front) continue;
				const size = (l.size * 1.7) / s.unitsPerPixel(q.depth);
				if (size < 6) continue;
				out.push({
					text: l.text(),
					x: q.x,
					y: q.y,
					size: Math.min(size, 40),
					alpha: 0.8,
					align: l.align,
					italic: l.italic
				});
			}
		}
		const near = (name: string) => this.stations[this.station]?.name === name;
		// the prompt's tokens under their shares
		if (near('Where it looks') && this.run) {
			const tc = this.textCell;
			for (let k = 0; k < this.textCols; k++) {
				const word = clean(this.run.tokens[k]) || '·';
				const q = s.project([this.textAt[0] + (k + 0.5) * tc, this.textAt[1] - 6.35, 0]);
				const upp = s.unitsPerPixel(q.depth);
				const size = Math.min(0.3 / upp, (tc * 0.95) / upp / Math.max(1, word.length * 0.5));
				if (q.front && size >= 5)
					out.push({
						text: word,
						x: q.x,
						y: q.y,
						size: Math.min(size, 18),
						alpha: k === this.other ? 1 : this.run.words.includes(k) ? 0.7 : 0.35,
						align: 'center'
					});
			}
		}
		// rings: the row (if a patch) and the row spelled out (if a patch), on the maps and the pictures
		const ring = (at: V3, size: number, p: number, glyph: string) => {
			const c: V3 = [
				at[0] + ((p % GRID) + 0.5) * (size / GRID),
				at[1] - (Math.floor(p / GRID) + 0.5) * (size / GRID),
				0.02
			];
			const q = s.project(c);
			const px = size / GRID / s.unitsPerPixel(q.depth);
			if (q.front && px * GRID > 40)
				out.push({
					text: glyph,
					x: q.x,
					y: q.y,
					size: Math.max(9, Math.min(26, px * 2.6)),
					alpha: 0.95,
					align: 'center'
				});
		};
		const marks: [number, string][] = [];
		if (this.isPatch) marks.push([this.patch, '○']);
		if (this.other >= NT) marks.push([this.other - NT, '◇']);
		for (const [p, g] of marks) {
			for (const pic of this.pics) ring(pic.at, pic.size, p, g);
			if (near('Where it looks')) this.mapAt.forEach((at, i) => ring(at, this.mapSize[i], p, g));
		}
		// the heads' labels on their maps
		if (near('Where it looks') && this.cpu) {
			for (let h = 0; h < H; h++) {
				const at = this.mapAt[h];
				const q = s.project([at[0], at[1] + 0.3, 0]);
				const size = 0.26 / s.unitsPerPixel(q.depth);
				if (!q.front || size < 7) continue;
				let pic = 0;
				for (let j = 0; j < 1024; j++) pic += this.cpu.attn[h * NJ + NT + j];
				out.push({
					text: `head ${h + 1}: ${pc(pic)} on the picture`,
					x: q.x,
					y: q.y,
					size: Math.min(size, 16),
					alpha: h === this.head ? 1 : 0.6,
					align: 'left'
				});
			}
		}
		return out;
	}

	/** Where a world point falls on a picture or a map: the patch, or null. */
	private patchUnder(p: V3): { patch: number; map: number } | null {
		const areas: [V3, number, number][] = [
			...this.pics.map((pic) => [pic.at, pic.size, -1] as [V3, number, number]),
			...this.mapAt.map((at, i) => [at, this.mapSize[i], i] as [V3, number, number])
		];
		for (const [at, size, map] of areas) {
			const u = (p[0] - at[0]) / size,
				v = (at[1] - p[1]) / size;
			if (u >= 0 && u < 1 && v >= 0 && v < 1)
				return { patch: Math.floor(v * GRID) * GRID + Math.floor(u * GRID), map };
		}
		return null;
	}

	/** Click: a row of a matrix to spell out, a word or a patch to compare with, or a patch to follow. */
	click(clientX: number, clientY: number): 'row' | null {
		const s = this.stage;
		const pt = s.pointUnder(clientX, clientY);
		if (pt) {
			const u = this.patchUnder(pt);
			if (u) {
				const p = this.nearestFollowed(u.patch);
				if (u.map < 0) {
					if (NT + p !== this.row) void this.choose({ row: NT + p });
					return 'row';
				}
				if (NT + p !== this.row) this.setDot(u.map < H ? u.map : this.head, NT + p);
				return null;
			}
		}
		const hit = this.panels.pick(s.ray(clientX, clientY));
		if (!hit) return null;
		const m = this.mats.get(hit.panel);
		if (m) {
			const key =
				m.key === 'q' ? 'qkv' : m.key === 'o' || m.key === 'down' || m.key === 'out' ? m.key : '';
			if (key) {
				this.rowPick.set(key, hit.row);
				void this.pickRows();
			}
			return null;
		}
		if (hit.panel === this.p.shares_t && this.run?.words.includes(hit.col) && hit.col !== this.row)
			this.setDot(hit.row, hit.col);
		return null;
	}

	/** What is under the pointer. */
	inspect(clientX: number, clientY: number): { title: string; lines: string[] } | null {
		const cpu = this.cpu;
		if (!cpu) return null;
		const s = this.stage;
		const pt = s.pointUnder(clientX, clientY);
		if (pt) {
			const u = this.patchUnder(pt);
			if (u) {
				const p = u.patch;
				const where = `row ${Math.floor(p / GRID) + 1}, column ${(p % GRID) + 1}`;
				if (u.map < 0)
					return {
						title: `the patch at ${where}`,
						lines: ['click to follow the nearest kept patch']
					};
				const share =
					u.map < H
						? cpu.attn[u.map * NJ + NT + p]
						: Array.from({ length: H }, (_, h) => cpu.attn[h * NJ + NT + p]).reduce(
								(a, b) => a + b,
								0
							) / H;
				return {
					title:
						u.map < H
							? `head ${u.map + 1}'s share for the patch at ${where}`
							: `the heads' mean share for ${where}`,
					lines: [fmt(share), 'click to compare with the nearest kept patch']
				};
			}
		}
		const hit = this.panels.pick(s.ray(clientX, clientY));
		if (!hit) return null;
		const m = this.mats.get(hit.panel);
		if (m)
			return {
				title: `${m.key}, block ${this.block + 1}`,
				lines: [`row ${hit.row}, column ${hit.col}`, 'click to spell this row out']
			};
		if (hit.panel === this.p.shares_t || hit.panel === this.p.shares_pad) {
			const col = hit.panel === this.p.shares_t ? hit.col : this.textCols + hit.col;
			return {
				title: `head ${hit.row + 1}'s share for ${col < this.textCols ? `“${this.tokenText(col)}”` : `padding row ${col}`}`,
				lines: [fmt(cpu.attn[hit.row * NJ + col])]
			};
		}
		return null;
	}

	destroy() {
		this.panels.destroy();
		this.prefixUniforms.destroy();
	}
}

function cap1(s: string) {
	return s.charAt(0).toUpperCase() + s.slice(1);
}

function pc(v: number | undefined) {
	if (v === undefined) return '';
	return `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
}
